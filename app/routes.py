from app.services.user_settings.user_settings_service import user_settings_service
from app.services.prediction.prediction_service import prediction_service
from app.services.authentication.auth_service import auth_service
from app.services.chatbot.chatbot_service import chatbot_service
from app.services.report.report_service import report_service
from app.core.managers.database_manager import db_manager
from werkzeug.utils import secure_filename
from app.models.user.user import User
from flask import send_file
from pathlib import Path
import os
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)

# Blueprint for main/public routes
main_bp = Blueprint("main", __name__)

# -------------------------------------------------------------------
# Upload configuration for brain MRI images
# -------------------------------------------------------------------
# Upload directory: app/ui/uploads/brain
# Flask static_folder is set to "ui", so url_for("static", filename="uploads/brain/...")
# will look for files in app/ui/uploads/brain/
# routes.py is in app/, so Path(__file__).parent is app/
BASE_DIR = Path(__file__).parent  # app/
BRAIN_UPLOAD_DIR = BASE_DIR / "ui" / "static" / "uploads" / "brain"
BRAIN_UPLOAD_DIR.mkdir(parents = True, exist_ok = True)

# Allowed MRI image extensions
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@main_bp.route("/") # Welcome / Home page
def welcome():
    return render_template("welcome.html")


@main_bp.route("/register", methods = ["GET", "POST"])  # Registration page.
def register():
    # GET: show the form
    if request.method == "POST":  # POST: process the form and create new user.
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Basic validation
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("main.register"))

        # Email format validation
        if "@" not in email or "." not in email.split("@")[-1]:
            flash("Please enter a valid email address.", "error")
            return redirect(url_for("main.register"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("main.register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return redirect(url_for("main.register"))

        try:
            success, message = auth_service.register(username, email, password)
            if success:
                flash(message, "success")
                return redirect(url_for("main.login"))
            else:
                flash(message, "error")
                return redirect(url_for("main.register"))
        except Exception as e:
            print(f"[ERROR] Register route: Unexpected error: {e}")
            flash("Registration failed. Please try again later.", "error")
            return redirect(url_for("main.register"))

    # GET request
    return render_template("register.html")


@main_bp.route("/login", methods = ["GET", "POST"])  # Login page
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")

        if not identifier or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("main.login"))

        success, result = auth_service.login(identifier, password)
        if not success:
            flash(result, "error")  # result is an error message
            return redirect(url_for("main.login"))

        user = result  # result is a User instance
        # Save minimal info in session
        session["user_id"] = user.id
        session["username"] = user.username

        flash(f"Welcome, {user.username}!", "success")
        return redirect(url_for("main.dashboard"))

    # GET request
    return render_template("login.html")


@main_bp.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access the dashboard.", "error")
        return redirect(url_for("main.login"))

    user_id = session.get("user_id")
    username = session.get("username")
    
    # Fetch statistics from prediction_logs
    try:
        # Total analyses (all predictions)
        total_row = db_manager.fetch_one(
            "SELECT COUNT(*) as count FROM prediction_logs WHERE user_id = ?",
            (user_id,)
        )
        total_analyses = total_row[0] if total_row else 0
        
        # Heart scans count
        heart_row = db_manager.fetch_one(
            "SELECT COUNT(*) as count FROM prediction_logs WHERE user_id = ? AND model_type = ?",
            (user_id, "heart_disease")
        )
        heart_scans = heart_row[0] if heart_row else 0
        
        # Brain scans count
        brain_row = db_manager.fetch_one(
            "SELECT COUNT(*) as count FROM prediction_logs WHERE user_id = ? AND model_type = ?",
            (user_id, "brain_tumor_multiclass")
        )
        brain_scans = brain_row[0] if brain_row else 0
    except Exception as e:
        print(f"[ERROR] Dashboard: Failed to fetch statistics: {e}")
        total_analyses = 0
        heart_scans = 0
        brain_scans = 0
    
    return render_template(
        "dashboard.html",
        username=username,
        total_analyses=total_analyses,
        heart_scans=heart_scans,
        brain_scans=brain_scans
    )


@main_bp.route("/heart-disease", methods = ["GET", "POST"])
def heart_disease():
    if "user_id" not in session:
        flash("Please log in to access heart disease detection.", "error")
        return redirect(url_for("main.login"))
    
    result = None
    if request.method == "POST":
        # UPDATED: These fields match the 70k Cardiovascular Dataset
        required_fields = [
            "age", "sex", "height", "weight", 
            "ap_hi", "ap_lo", "cholesterol", "gluc", 
            "smoke", "alco", "active"
        ]
        
        # Validate all required fields are present and not empty
        missing_fields = []
        for field in required_fields:
            value = request.form.get(field, "").strip()
            if not value:
                missing_fields.append(field)
        
        if missing_fields:
            flash(f"Please fill in all required fields. Missing: {', '.join(missing_fields)}", "error")
            return redirect(url_for("main.heart_disease"))
        
        # Validate numeric fields are valid numbers
        numeric_fields = ["age", "height", "weight", "ap_hi", "ap_lo"]
        for field in numeric_fields:
            try:
                value = float(request.form.get(field, ""))
                if value < 0:
                    flash(f"{field} must be a positive number.", "error")
                    return redirect(url_for("main.heart_disease"))
            except ValueError:
                flash(f"{field} must be a valid number.", "error")
                return redirect(url_for("main.heart_disease"))
        
        user_id = session.get("user_id")
        form_data = request.form.to_dict()
        
        try:
            result = prediction_service.predict_heart_disease(form_data, user_id)
            flash("Heart prediction completed.", "success")
        except RuntimeError as e:
            flash(str(e), "error")
            return redirect(url_for("main.heart_disease"))
        except Exception as e:
            print(f"[ERROR] Heart disease route: Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            flash("Heart disease prediction failed. Please try again later.", "error")
            return redirect(url_for("main.heart_disease"))
    
    return render_template("heart_disease.html", result = result)

@main_bp.route("/brain-tumor", methods = ["GET", "POST"])
def brain_tumor():  # Brain tumor detection page
    if "user_id" not in session:
        flash("Please log in to access brain tumor detection.", "error")
        return redirect(url_for("main.login"))

    result = None
    if request.method == "POST":
        file = request.files.get("mri_image")

        # Validate file was uploaded
        if not file:
            flash("Please select an MRI image to upload.", "error")
            return redirect(url_for("main.brain_tumor"))

        # Validate filename is not empty
        if file.filename == "" or not file.filename:
            flash("Please select a valid image file.", "error")
            return redirect(url_for("main.brain_tumor"))

        # Secure the filename
        filename = secure_filename(file.filename)
        if not filename:
            flash("Invalid filename. Please select a valid image file.", "error")
            return redirect(url_for("main.brain_tumor"))

        ext = Path(filename).suffix.lower()

        # Validate file extension
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            flash(
                "Unsupported file type. Please upload a PNG, JPG, JPEG, or BMP image.",
                "error",
            )
            return redirect(url_for("main.brain_tumor"))

        # Full save path: app/ui/uploads/brain/<filename>
        # Flask static_folder is "ui", so url_for("static", filename="uploads/brain/...") 
        # looks for files in app/ui/uploads/brain/
        # Use os.path.join to ensure proper path handling on all platforms
        save_path = os.path.join(str(BRAIN_UPLOAD_DIR), filename)

        # Save the file
        try:
            file.save(save_path)
        except Exception as e:
            print(f"[ERROR] Failed to save uploaded MRI: {e}")
            flash("There was a problem saving the uploaded image. Please try again.", "error")
            return redirect(url_for("main.brain_tumor"))

        # Build the public URL for the uploaded image using static folder
        # Use forward slashes for URL (works on all platforms)
        image_url = url_for("static", filename = f"uploads/brain/{filename}")
        user_id = session.get("user_id")

        # Run prediction
        try:
            # Pass string path to prediction service
            result = prediction_service.predict_brain_tumor(str(save_path), user_id)
            # Add image URL to result for template display
            if result:
                result["image_url"] = image_url
            flash(
                "Brain tumor prediction completed successfully – see results below.",
                "success",
            )
        except RuntimeError as e:
            # RuntimeError messages are user-friendly
            flash(str(e), "error")
            result = None
        except Exception as e:
            print(f"[ERROR] Brain tumor prediction failed: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            flash("Brain tumor prediction failed. Please try again later.", "error")
            result = None

    return render_template("brain_tumor.html", result = result)

@main_bp.route("/chatbot", methods = ["GET", "POST"])
def chatbot():  # AI Doctor Chatbot page
    if "user_id" not in session:
        flash("Please log in to access the AI doctor chatbot.", "error")
        return redirect(url_for("main.login"))

    user_message = None
    assistant_reply = None

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()

        if not user_message:
            flash("Please type a message before sending.", "error")
            return redirect(url_for("main.chatbot"))

        user_id = session.get("user_id")

        try:
            assistant_reply = chatbot_service.send_message(user_id, user_message)
        except RuntimeError as e:
            # Handle specific errors like missing API key
            error_msg = str(e)
            if "GROQ_API_KEY" in error_msg:
                flash("Chatbot configuration error: GROQ_API_KEY environment variable is not set. Please set the GROQ_API_KEY environment variable before using the chatbot. You can get an API key from https://console.groq.com/", "error")
            else:
                flash(f"Chatbot error: {error_msg}", "error")
            print(f"[ERROR] ChatbotService failed: {e}")
            assistant_reply = None
        except Exception as e:
            # Handle other unexpected errors
            error_msg = str(e)
            flash("There was a problem contacting the AI doctor chatbot. Please try again later.", "error")
            print(f"[ERROR] ChatbotService failed: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            assistant_reply = None

    return render_template(
        "chatbot.html",
        user_message = user_message,
        assistant_reply = assistant_reply,
    )

@main_bp.route("/settings", methods=["GET"])
def settings():
    """
    Settings & Profile page:
    - Shows profile info
    - Links/forms for password change, history clear, delete account
    """
    if "user_id" not in session:
        flash("Please log in to access settings.", "error")
        return redirect(url_for("main.login"))

    user_id = session.get("user_id")
    profile = user_settings_service.get_profile(user_id)

    if profile is None:
        flash("User not found.", "error")
        return redirect(url_for("main.dashboard"))

    return render_template("settings.html", profile = profile)

@main_bp.route("/settings/change-password", methods = ["POST"])
def change_password():
    if "user_id" not in session:
        flash("Please log in to access settings.", "error")
        return redirect(url_for("main.login"))

    user_id = session.get("user_id")
    old_password = request.form.get("old_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    success, message = user_settings_service.change_password(
        user_id,
        old_password,
        new_password,
        confirm_password,
    )

    flash(message, "success" if success else "error")
    return redirect(url_for("main.settings"))

@main_bp.route("/settings/clear-history", methods=["POST"])
def clear_history():
    if "user_id" not in session:
        flash("Please log in to access settings.", "error")
        return redirect(url_for("main.login"))

    user_id = session.get("user_id")

    success, message = user_settings_service.clear_prediction_history(user_id)
    flash(message, "success" if success else "error")
    return redirect(url_for("main.settings"))

@main_bp.route("/settings/delete-account", methods=["POST"])
def delete_account():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("main.login"))

    user_id = session.get("user_id")
    success, message = user_settings_service.delete_account(user_id)
    session.clear()

    flash(message, "success" if success else "error")
    return redirect(url_for("main.welcome"))

@main_bp.route("/reports/heart/<int:log_id>")
def heart_report(log_id: int):
    """
    Download a PDF report for a heart-disease prediction.
    """
    if "user_id" not in session:
        flash("Please log in to access reports.", "error")
        return redirect(url_for("main.login"))

    # Validate log_id
    if not log_id or log_id <= 0:
        flash("Invalid report ID.", "error")
        return redirect(url_for("main.dashboard"))

    user_id = session.get("user_id")

    try:
        # Get the prediction log and make sure it belongs to this user
        log = report_service.get_prediction_for_user(
            log_id,
            user_id,
            model_type = "heart_disease",
        )
        if log is None:
            flash("Heart prediction log not found.", "error")
            return redirect(url_for("main.dashboard"))

        # Load user object
        row = db_manager.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if row is None:
            flash("User not found.", "error")
            return redirect(url_for("main.dashboard"))

        user = User.from_row(row)

        # Generate PDF
        try:
            pdf_buffer = report_service.generate_heart_report(user, log)
        except Exception as e:
            print(f"[ERROR] Heart report PDF generation failed: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            flash("Could not generate PDF report. Please try again later.", "error")
            return redirect(url_for("main.dashboard"))

        filename = f"heart report {log.get('id')}.pdf"
        return send_file(
            pdf_buffer,
            mimetype = "application/pdf",
            as_attachment = True,
            download_name = filename,
        )
    except Exception as e:
        print(f"[ERROR] Heart report route: Unexpected error: {type(e).__name__}: {e}")
        flash("An error occurred while generating the report. Please try again later.", "error")
        return redirect(url_for("main.dashboard"))

@main_bp.route("/reports/brain/<int:log_id>")
def brain_report(log_id: int):
    """
    Download a PDF report for a brain-tumor prediction.
    """
    if "user_id" not in session:
        flash("Please log in to access reports.", "error")
        return redirect(url_for("main.login"))

    # Validate log_id
    if not log_id or log_id <= 0:
        flash("Invalid report ID.", "error")
        return redirect(url_for("main.dashboard"))

    user_id = session.get("user_id")

    try:
        # Get the prediction log and make sure it belongs to this user
        log = report_service.get_prediction_for_user(
            log_id,
            user_id,
            model_type = "brain_tumor_multiclass",
        )
        if log is None:
            flash("Brain prediction log not found.", "error")
            return redirect(url_for("main.dashboard"))

        # Load user object
        row = db_manager.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if row is None:
            flash("User not found.", "error")
            return redirect(url_for("main.dashboard"))

        user = User.from_row(row)

        # Generate PDF
        try:
            pdf_buffer = report_service.generate_brain_report(user, log)
        except Exception as e:
            print(f"[ERROR] Brain report PDF generation failed: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            flash("Could not generate PDF report. Please try again later.", "error")
            return redirect(url_for("main.dashboard"))

        filename = f"brain report {log.get('id')}.pdf"
        return send_file(
            pdf_buffer,
            mimetype = "application/pdf",
            as_attachment = True,
            download_name = filename,
        )
    except Exception as e:
        print(f"[ERROR] Brain report route: Unexpected error: {type(e).__name__}: {e}")
        flash("An error occurred while generating the report. Please try again later.", "error")
        return redirect(url_for("main.dashboard"))
    
@main_bp.route("/logout")  # Log the user out by clearing the session
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("main.welcome"))
