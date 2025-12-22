from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.core.managers.database_manager import db_manager, DatabaseManager
from app.core.managers.model_manager import model_manager
from app.services.base_service import BaseService


class PredictionService(BaseService):    # Handles prediction logic for heart disease and brain tumor
    # Uses ModelManager to access models and DatabaseManager to log results
    def __init__(self, db: DatabaseManager = db_manager) -> None:
        super().__init__(db)
        # Preserve original public attributes
        self.db = self._db
        self.models = model_manager
        
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    def _parse_float(self, value: str, default: float = 0.0) -> float: 
        # Safely parse a string to float. Returns default if parsing fails
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def predict_heart_disease(self, form_data: Dict[str, str], user_id: Optional[int]) -> Dict[str, Any]:
        """
        Take raw form data -> build feature dict -> call HeartDiseaseModel ->
        log prediction (if user_id) -> return structured result + log_id.
        Raises RuntimeError if model fails or prediction fails.
        """

        try:
            # --- 1. Parse Inputs for 70k Dataset ---
            
            # AGE: Dataset uses days. User inputs years.
            age_years = self._parse_float(form_data.get("age"))
            age_days = age_years * 365

            # GENDER: 1 = Female, 2 = Male (Standard for this specific dataset)
            # We assume form sends "1" for Male, "0" for Female. We must map to 2/1.
            sex_input = form_data.get("sex", "1") # Default male
            if str(sex_input) == "1": 
                gender = 2.0 # Male
            else:
                gender = 1.0 # Female

            height = self._parse_float(form_data.get("height")) # cm
            weight = self._parse_float(form_data.get("weight")) # kg
            ap_hi = self._parse_float(form_data.get("ap_hi"))   # Systolic
            ap_lo = self._parse_float(form_data.get("ap_lo"))   # Diastolic

            # Categoricals (1=Normal, 2=Above Normal, 3=Well Above)
            cholesterol = self._parse_float(form_data.get("cholesterol"), 1.0)
            gluc = self._parse_float(form_data.get("gluc"), 1.0)

            # Binary (0 or 1)
            smoke = self._parse_float(form_data.get("smoke"), 0.0)
            alco = self._parse_float(form_data.get("alco"), 0.0)
            active = self._parse_float(form_data.get("active"), 0.0)
            
            # 2. CALCULATE BMI (New Feature)
            if height > 0:
                bmi = weight / ((height / 100) ** 2)
            else:
                bmi = 25.0 # Default fallback
            
            # Build features dict (keys must match training script)
            features = {
                "age": age_days,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            }

            # Convert to list for model
            feature_values = list(features.values())
            
            # --------------------
            # 2) Predict
            # --------------------
            try:
                heart_model = self.models.get_heart_model()
                risk_label, probability = heart_model.predict(features)
            except RuntimeError as e:
                raise RuntimeError(f"Heart disease model error: {str(e)}")
            except Exception as e:
                print(f"[ERROR] PredictionService.predict_heart_disease: Model prediction failed: {e}")
                raise RuntimeError("Heart disease prediction failed. Please try again later.")

            # Short summary for DB
            input_summary = (f"Age:{age_years}, H:{height}, W:{weight}, BP:{ap_hi}/{ap_lo}, "
                             f"Smoke:{smoke}, Active:{active}")

            # --------------------
            # 3) Log to DB + get log_id
            # --------------------
            log_id: Optional[int] = None

            if user_id is not None:
                try:
                    # Insert log and get the inserted row ID in the same transaction
                    log_id = self.db.execute_and_get_id(
                        """
                        INSERT INTO prediction_logs (
                            user_id, model_type, input_summary,
                            prediction_result, probability, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            "heart_disease",
                            input_summary,
                            risk_label,
                            float(probability),
                            self._now_iso(),
                        ),
                    )
                except Exception as e:
                    print(f"[ERROR] PredictionService.predict_heart_disease: Failed to log prediction: {e}")
                    # Continue without log_id if logging fails

            # --------------------
            # 4) Return result dict (used in templates)
            # --------------------
            return {
                "risk_label": risk_label,
                "probability": probability,
                "features": features,
                "input_summary": input_summary,
                "suggestion": self._generate_heart_suggestion(risk_label),
                "log_id": log_id,
            }
        except RuntimeError:
            # Re-raise RuntimeErrors as-is (they have user-friendly messages)
            raise
        except Exception as e:
            print(f"[ERROR] PredictionService.predict_heart_disease: Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise RuntimeError("An unexpected error occurred during heart disease prediction. Please try again.")

    def _generate_heart_suggestion(self, risk_label: str) -> str: # Treatment suggestion
        if risk_label == "High":
            return (
                "HIGH RISK INDICATED\n"
                "Recommended Actions:\n"
                "1. Consult a cardiologist immediately for a full evaluation.\n"
                "2. monitor your Blood Pressure daily.\n"
                "3. Adhere to a strict low-sodium, low-saturated fat diet.\n"
                "4. Avoid strenuous physical activity until cleared by a doctor.\n"
                "5. If you smoke or drink, stop immediately."
            )
        elif risk_label == "Medium":
            return (
                "MODERATE RISK - LIFESTYLE CHANGES REQUIRED\n"
                "Suggested Plan:\n"
                "1. Schedule a check-up with your Doctor within the next month.\n"
                "2. Adopt the DASH or Mediterranean diet (more veggies, less processed food).\n"
                "3. Aim for 30 minutes of moderate exercise (like walking) 5 days a week.\n"
                "4. Reduce stress through sleep (7-8 hours) and mindfulness."
            )
        else:
            return (
                "LOW RISK - MAINTENANCE MODE\n"
                "Keep up the good work:\n"
                "1. Continue your balanced diet and active lifestyle.\n"
                "2. Get an annual physical check-up to track changes.\n"
                "3. Stay hydrated and ensure consistent sleep quality.\n"
                "4. Avoid smoking to keep your risk low."
            )
    
    def predict_brain_tumor(self, image_path: str, user_id: Optional[int]) -> Dict[str, Any]:
        """
        Take an MRI image path -> call BrainTumorModel -> log prediction -> return result.
        Raises RuntimeError if model fails or prediction fails.
        """
        try:
            # Get brain model
            try:
                brain_model = self.models.get_brain_model()
            except RuntimeError as e:
                raise RuntimeError(f"Brain tumor model error: {str(e)}")
            
            # Run prediction
            try:
                model_result = brain_model.predict(image_path)
            except FileNotFoundError as e:
                print(f"[ERROR] PredictionService.predict_brain_tumor: Image file not found: {e}")
                raise RuntimeError("Image file not found. Please ensure the file was uploaded correctly.")
            except ValueError as e:
                print(f"[ERROR] PredictionService.predict_brain_tumor: Image preprocessing error: {e}")
                raise RuntimeError(f"Error processing image: {str(e)}. Please ensure you're uploading a valid image file.")
            except Exception as e:
                print(f"[ERROR] PredictionService.predict_brain_tumor: Model prediction failed: {e}")
                raise RuntimeError("Brain tumor prediction failed. Please try again later.")
            
            predicted_class: str = model_result.get("predicted_class", "unknown")
            probability: float = float(model_result.get("probability", 0.0))
            probabilities: Dict[str, float] = model_result.get("probabilities", {})
            
            # Decide if this is considered "tumor" or "no_tumor"
            tumor_classes = {"glioma", "meningioma", "pituitary"}
            is_tumor = predicted_class in tumor_classes
            
            # Build a short input summary for logging
            input_summary = f"image_path={image_path.name if hasattr(image_path, 'name') else str(image_path)}"

            # --------------------
            # Log prediction in DB + get log_id
            # --------------------
            log_id: Optional[int] = None

            if user_id is not None:
                try:
                    # Insert log and get the inserted row ID in the same transaction
                    log_id = self.db.execute_and_get_id(
                        """
                        INSERT INTO prediction_logs (
                            user_id, model_type, input_summary,
                            prediction_result, probability, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            "brain_tumor_multiclass",
                            input_summary,
                            predicted_class,
                            probability,
                            self._now_iso(),
                        ),
                    )
                except Exception as e:
                    print(f"[ERROR] PredictionService.predict_brain_tumor: Failed to log prediction: {e}")
                    # Continue without log_id if logging fails

            # Build a user-friendly suggestion message
            suggestion = self._generate_brain_suggestion(predicted_class, is_tumor, probability)

            return {
                "predicted_class": predicted_class,
                "probability": probability,
                "probabilities": probabilities,
                "is_tumor": is_tumor,
                "input_summary": input_summary,
                "suggestion": suggestion,
                "log_id": log_id,
            }
        except RuntimeError:
            # Re-raise RuntimeErrors as-is (they have user-friendly messages)
            raise
        except Exception as e:
            print(f"[ERROR] PredictionService.predict_brain_tumor: Unexpected error: {type(e).__name__}: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise RuntimeError("An unexpected error occurred during brain tumor prediction. Please try again.")

    def _generate_brain_suggestion(
        self,
        predicted_class: str,
        is_tumor: bool,
        probability: float,
    ) -> str:

        prob_pct = round(probability * 100)

        if predicted_class == "no_tumor":
            return (
                f"The model's highest confidence class is 'no_tumor' "
                f"with an estimated probability of about {prob_pct}%. "
                "This does not guarantee that no abnormality exists. "
                "If you have any symptoms or concerns, please consult a neurologist or radiologist."
            )

        # tumor classes
        base_msg = (
            f"The model suggests the MRI is most consistent with '{predicted_class}' "
            f"with an estimated probability of about {prob_pct}%. "
            "This is NOT a clinical diagnosis."
        )

        follow_up = (
            "You should promptly consult a qualified neurologist or neurosurgeon, "
            "and have this MRI evaluated by a radiologist for a professional interpretation."
        )

        return base_msg + " " + follow_up
    
# Global instance used by routes
prediction_service = PredictionService()
