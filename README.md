# 🏥 Multi Disease Detection System (MDDS)

**An Integrated Healthcare Platform powered by Machine Learning & AI**

![Project Banner](https://via.placeholder.com/1200x400.png?text=Multi+Disease+Detection+System+Dashboard)
*(Note: Replace the image link above with a real screenshot of your new medical dashboard)*

## 📋 Overview

The **Multi Disease Detection System** is a robust web-based application designed to assist in the early detection and risk assessment of critical health conditions. By combining traditional **Machine Learning** (for heart disease) and **Deep Learning** (for brain tumor MRI analysis) with a state-of-the-art **AI Medical Chatbot**, MDDS provides users with a holistic tool for personal health monitoring.

The system features a professional, medical-grade user interface, secure authentication, and detailed PDF reporting capabilities.

---

## ✨ Key Features

### 1. ❤️ Heart Disease Risk Assessment
* **Algorithm:** Random Forest Classifier.
* **Input Data:** Clinical parameters including Age, Gender, BMI (derived from Height/Weight), Blood Pressure (Systolic/Diastolic), Cholesterol, and Glucose levels.
* **Output:** Risk classification (Low, Medium, High) with a probability confidence score and tailored medical suggestions.

### 2. 🧠 Brain Tumor Detection
* **Algorithm:** Convolutional Neural Network (CNN) built with TensorFlow/Keras.
* **Input Data:** MRI Scans (Image Upload).
* **Classes Detected:**
    * Glioma Tumor
    * Meningioma Tumor
    * Pituitary Tumor
    * No Tumor
* **Output:** Tumor classification with confidence percentages for all classes.

### 3. 🤖 Dr. MDDS (AI Medical Assistant)
* **Powered By:** Groq Cloud API (Llama-3.1-8b-instant).
* **Context-Aware:** The chatbot has "memory" of your latest screening results. It can connect your current symptoms (e.g., "headache") to your specific test history (e.g., "brain tumor analysis").
* **Capabilities:**
    * Symptom Triage & Analysis.
    * Medication Guidance (Educational info on interactions/side effects).
    * Strict safety protocols to prevent non-medical discussions.

### 4. 📄 Reporting & User Management
* **PDF Reports:** Download detailed medical reports for any analysis performed.
* **User Dashboard:** Track history of past predictions.
* **Security:** Secure login/registration with password hashing.

---

## 🛠️ Tech Stack

* **Backend Framework:** Python (Flask)
* **Database:** SQLite (Lightweight, file-based)
* **Machine Learning:**
    * Scikit-learn (Random Forest)
    * TensorFlow / Keras (CNN)
    * Joblib (Model serialization)
    * NumPy / Pandas (Data processing)
* **AI / LLM:** Groq API (Llama 3.1)
* **Frontend:** HTML5, CSS3 (Medical Theme), JavaScript
* **Version Control:** Git & Git LFS (Large File Storage)

---

## 📂 Project Structure

```text
Multi-Disease-Detection-System/
├── app/
│   ├── core/                   # Core managers (DB, Models)
│   ├── data/                   # Datasets and Saved Models (.pkl, .h5)
│   ├── models/                 # Data classes (User, Heart, Brain)
│   ├── services/               # Business logic (Auth, Prediction, Chatbot)
│   ├── ui/                     # Frontend templates (HTML) and static files (CSS/JS)
│   ├── routes.py               # Flask route definitions
│   └── __init__.py             # App factory
├── instance/                   # SQLite database file
├── model_training/             # Scripts to train ML models from scratch
├── .env                        # Environment variables (API Keys) - NOT tracked by Git
├── .gitattributes              # Git LFS configuration
├── requirements.txt            # Python dependencies
└── run.py                      # Application entry point
```
## 📖 Usage Guide

1.  **Register/Login:** Create an account to access the features.
2.  **Heart Check:** Navigate to "Heart Disease" from the sidebar. Enter your clinical details (BP, age, etc.) and click "Run Assessment."
3.  **Brain Scan:** Go to "Brain MRI." Upload a clear JPG/PNG of a brain MRI. The system will analyze it for tumors.
4.  **Consult Dr. MDDS:** Open the "AI Doctor."
    * *Ask:* "I have a headache." (The bot will check if you recently had a brain scan).
    * *Ask:* "What is Panadol used for?" (The bot will provide educational medication info).
5.  **Settings:** Update your theme (Light/Dark) or manage your account data.

---

## 🤖 Model Details

### Heart Disease Model
* **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (70,000 records).
* **Features:** Age, Gender, Height, Weight, AP_Hi, AP_Lo, Cholesterol, Glucose, Smoke, Alcohol, Active.
* **Performance:** ~73% Accuracy (Random Forest).

### Brain Tumor Model
* **Dataset:** [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
* **Architecture:** Custom CNN with 4 Convolutional layers, Max Pooling, and Dropout for regularization.
* **Performance:** ~96%+ Accuracy on test set.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mahmoud Naguib**
* GitHub: [@MahmoudNagiubX](https://github.com/MahmoudNagiubX)
