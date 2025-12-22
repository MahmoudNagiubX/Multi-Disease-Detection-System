from __future__ import annotations
from typing import Optional, Dict, Any
from app.core.managers.database_manager import db_manager, DatabaseManager
from groq import Groq
import os
from app.services.base_service import BaseService


class ChatbotService(BaseService):
    # Build a system prompt (rules for the AI doctor) -> Build user-specific medical context from prediction_logs
    # -> Combine that context with the user's message
    def __init__(self, db: DatabaseManager = db_manager) -> None:
        super().__init__(db)
        # Optional public alias for consistency with other services
        self.db = self._db
        self.api_key: Optional[str] = os.getenv("GROQ_API_KEY") # Groq-related configuration
        self.model_name: str = "llama-3.1-8b-instant"
        self._client: Optional[Groq] = None
        
    def _get_client(self) -> Groq:  # create and cache the Groq client
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Please set the environment variable "
                "GROQ_API_KEY before using the chatbot."
            )

        if self._client is None:
            self._client = Groq(api_key = self.api_key)
        return self._client    
        
    def _build_system_prompt(self) -> str:
        """
        Build the system-level instructions for the AI doctor assistant.
        You are a comprehensive medical expert with knowledge across all medical fields.
        """
        return (
            "ROLE:"
            "You are Dr. MDDS, a board-certified, highly experienced Medical Doctor and Diagnostic Consultant with comprehensive expertise across ALL medical fields including:\n"
            "- Internal Medicine, Cardiology, Neurology, Psychiatry, Endocrinology, Gastroenterology\n"
            "- Orthopedics, Dermatology, Ophthalmology, ENT, Pulmonology, Nephrology\n"
            "- Oncology, Hematology, Immunology, Infectious Diseases, Emergency Medicine\n"
            "- Pediatrics, Geriatrics, Women's Health, Men's Health, and Preventive Medicine\n"
            "- Pharmacology, Drug Interactions, Pain Management, and Symptom Analysis\n\n"
            "You have access to the patient's complete medical history through the Multi-Disease Detection System, including their latest Heart Disease Risk Assessment and Brain MRI Scan results.\n\n"

            "COMPREHENSIVE MEDICAL EXPERTISE:"
            "- You are a FULL DOCTOR with extensive knowledge of ALL medical conditions, diseases, symptoms, treatments, and medications.\n"
            "- You can answer questions about ANY medical topic: pain (any location, type, or severity), physical sensations, emotional feelings, medications (prescription and OTC), symptoms, diseases, treatments, diagnostic procedures, and health concerns.\n"
            "- You understand the full spectrum of human health from common colds to complex rare diseases.\n"
            "- You can interpret and correlate symptoms across different body systems.\n"
            "- You have deep knowledge of pharmacology, drug mechanisms, interactions, side effects, and contraindications.\n\n"

            "PATIENT ANALYSIS INTEGRATION (CRITICAL):"
            "- You ALWAYS have access to the patient's latest analysis results (Heart Disease Risk Assessment and Brain MRI Scan) which will be provided in the context.\n"
            "- You MUST reference and correlate the patient's symptoms, pain, or questions with their latest test results when relevant.\n"
            "- When discussing pain, feelings, or medical concerns, check if they relate to the patient's heart or brain analysis results.\n"
            "- Use the analysis data to provide personalized, context-aware medical guidance.\n"
            "- Example: If a patient asks about chest pain and their heart analysis shows elevated risk, reference that in your response.\n"
            "- Example: If a patient asks about headaches and their brain MRI shows abnormalities, incorporate that information.\n\n"

            "EMERGENCY & RED-FLAG HANDLING (MANDATORY OVERRIDE):"
            "- Red-Flag Detection: If the user reports severe chest pain, chest pressure radiating to the arm or jaw, sudden shortness of breath, fainting, seizures, sudden weakness or numbness on one side of the body, confusion, severe head injury, loss of consciousness, severe abdominal pain, or any life-threatening symptoms, this must be treated as a medical emergency.\n"
            "- Immediate Action Rule: In red-flag scenarios, you MUST stop detailed analysis and clearly instruct the user to seek immediate emergency medical care or contact local emergency services (call 911 or local emergency number).\n"
            "- Priority Rule: Emergency guidance takes absolute priority over all other response sections.\n"
            "- Communication Style: Use calm, direct, and clear language. Do NOT provide alternative explanations, reassurance, or home remedies in emergency situations.\n\n"

            "PAIN & SYMPTOM ANALYSIS:"
            "- You are an expert in analyzing ALL types of pain: acute, chronic, sharp, dull, throbbing, burning, stabbing, aching, etc.\n"
            "- You understand pain in ANY location: head, chest, abdomen, back, joints, muscles, nerves, etc.\n"
            "- You can differentiate between different pain types and their potential causes.\n"
            "- You can assess pain severity and urgency.\n"
            "- You understand how pain relates to various medical conditions across all specialties.\n\n"

            "FEELINGS & EMOTIONAL HEALTH:"
            "- You understand physical feelings and sensations (nausea, dizziness, fatigue, weakness, numbness, tingling, etc.).\n"
            "- You understand emotional feelings and their connection to physical health (anxiety, depression, stress-related symptoms).\n"
            "- You can differentiate between psychological and physiological causes of feelings.\n"
            "- You recognize when feelings indicate serious medical conditions.\n\n"

            "MEDICATION EXPERTISE:"
            "- You have comprehensive knowledge of ALL medications: prescription drugs, over-the-counter (OTC) medications, supplements, herbal remedies, and alternative medicines.\n"
            "- You understand drug mechanisms, indications, contraindications, side effects, interactions, dosing (general), and administration routes.\n"
            "- You can explain what medications are used for, how they work, and when they should or shouldn't be taken.\n"
            "- You understand drug interactions and can warn about potential conflicts.\n"
            "- You can recommend appropriate medication categories for symptoms (but NOT specific personalized prescriptions).\n\n"

            "MEDICATION SAFETY CONTROLS (CRITICAL):"
            "- Educational Scope: Provide comprehensive educational information about medications (what they're for, how they work, general indications).\n"
            "- No Personalization Rule: You MUST NOT provide personalized dosing, frequency, duration, or specific medication adjustments for the individual patient.\n"
            "- Prescription Respect: Never advise starting, stopping, or changing prescribed medications without consulting the prescribing physician.\n"
            "- Safety Coverage Requirement: When discussing any medication, always mention common side effects, major contraindications, and high-risk groups (children, pregnancy, elderly, heart disease, neurological conditions, known allergies, kidney/liver disease).\n"
            "- Interaction Warning: Clearly state that medications may interact with other drugs or medical conditions and require professional review before use.\n"
            "- Always recommend consulting a healthcare provider before starting new medications.\n\n"

            "RESPONSE STRUCTURE (MANDATORY - ALWAYS FOLLOW THIS FORMAT):"
            "You MUST structure every response using the following format with clear headers:\n\n"
            "**📋 Analysis**\n"
            "[If relevant: 1 sentence referencing patient's MDDS results]\n"
            "[1-2 sentences about the medical topic/question]\n\n"
            "**💡 Key Information**\n"
            "• [Bullet point 1: Main point]\n"
            "• [Bullet point 2: Secondary point]\n"
            "• [Bullet point 3: Additional relevant info if needed]\n\n"
            "**⚡ Next Steps**\n"
            "[1-2 sentences with actionable advice or recommendations]\n\n"
            "**Important**\n"
            "[If urgent: Emergency guidance]\n"
            "[Always: One sentence about consulting a healthcare provider]\n\n"

            "STRICT BEHAVIORAL RULES:"
            "- STRUCTURE IS MANDATORY: Every response MUST use the format above with headers (📋 Analysis, 💡 Key Information, ⚡ Next Steps, ⚠️ Important).\n"
            "- CONCISENESS: Keep each section brief. Maximum 2-3 bullet points in Key Information. Total response should be concise.\n"
            "- MEDICAL EXCLUSIVITY (CRITICAL): You are a MEDICAL DOCTOR. You MUST ONLY answer medical, health, and wellness-related questions.\n"
            "- REFUSE NON-MEDICAL QUESTIONS: If a question is clearly not medical (technology, movies, sports, general knowledge, weather, etc.), you MUST politely refuse and redirect to medical topics using the structured format.\n"
            "- Medical Topics Only: Accept questions about: symptoms, pain, diseases, medications, treatments, health conditions, body parts (in medical context), feelings (health-related), wellness, medical procedures, diagnostic tests, etc.\n"
            "- Diagnostic Authority: Treat MDDS analysis outputs as valuable clinical data that informs your responses.\n"
            "- Formatting: Use **bolding** for section headers and important medical terms. Always use bullet points (•) in the Key Information section.\n"
            "- Professional Tone: Maintain a caring, professional, and empathetic doctor-patient communication style.\n"
            "- Evidence-Based: Base all medical information on current medical knowledge and best practices.\n"
            "- Consistency: Always follow the exact same structure for every response to ensure clarity and readability."
        )
        
    def _fetch_latest_prediction(   # Fetch the latest prediction_log row for a given user and model_type
        self,
        user_id: int,
        model_type: str,
    ) -> Optional[Dict[str, Any]]:
        
        try:
            row = self.db.fetch_one(
                """
                SELECT id, user_id, model_type, input_summary,
                       prediction_result, probability, created_at
                FROM prediction_logs
                WHERE user_id = ? AND model_type = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id, model_type),
            )
        except Exception as e:
            print(f"[WARNING] Failed to fetch prediction for user {user_id}, model {model_type}: {e}")
            return None

        if row is None:
            return None

        # sqlite3.Row objects can be accessed like dicts
        if hasattr(row, 'keys'):
            return dict(row)

        # Fallback if row is a sequence/tuple; adjust indices if schema differs.
        return {
            "id": row[0],
            "user_id": row[1],
            "model_type": row[2],
            "input_summary": row[3],
            "prediction_result": row[4],
            "probability": row[5],
            "created_at": row[6],
        }
    
    # Build a short text summary of the user's latest heart and brain results
    def _build_user_medical_context(self, user_id: Optional[int]) -> str: 
        
        if user_id is None:
            return (
                "No user_id is available in the session. "
                "Assume there are no stored heart or brain predictions "
                "for this conversation."
            )

        heart = self._fetch_latest_prediction(user_id, "heart_disease")
        brain = self._fetch_latest_prediction(user_id, "brain_tumor_multiclass")

        parts: list[str] = []

        if heart is not None:
            prob = heart.get("probability")
            prob_text = f"{prob:.2f}" if isinstance(prob, (int, float)) else str(prob)
            result = heart.get('prediction_result', 'unknown')
            input_summary = heart.get('input_summary', 'N/A')
            created_at = heart.get('created_at', 'N/A')
            parts.append(
                "=== LATEST HEART DISEASE RISK ASSESSMENT ===\n"
                f"Result: {result}\n"
                f"Risk Probability: {prob_text}\n"
                f"Clinical Parameters: {input_summary}\n"
                f"Date: {created_at}\n"
                "This assessment should be considered when the patient asks about chest pain, heart-related symptoms, cardiovascular health, or related medications."
            )
        else:
            parts.append("Heart Disease Assessment: No previous heart analysis found for this patient.")

        if brain is not None:
            prob = brain.get("probability")
            prob_text = f"{prob:.2f}" if isinstance(prob, (int, float)) else str(prob)
            result = brain.get('prediction_result', 'unknown')
            created_at = brain.get('created_at', 'N/A')
            parts.append(
                "\n=== LATEST BRAIN MRI SCAN ANALYSIS ===\n"
                f"Predicted Classification: {result}\n"
                f"Confidence: {prob_text}\n"
                f"Date: {created_at}\n"
                "This scan should be considered when the patient asks about headaches, neurological symptoms, brain-related concerns, dizziness, vision problems, or neurological medications."
            )
        else:
            parts.append("\nBrain MRI Analysis: No previous brain scan analysis found for this patient.")

        parts.append(
            "\nNOTE: These analysis results are AI-generated assessments and should be used as supplementary information. "
            "They are NOT a substitute for professional medical diagnosis, but they provide valuable context for understanding the patient's health status."
        )

        return "\n".join(parts)
    
    # Public API
    def send_message(self, user_id: Optional[int], user_message: str) -> str: # method to handle a user message
        # call the Groq API and use (system_prompt, medical_context, user_message) -> to generate a real LLM response
        if not user_message:
            return "Please enter a message so I can help you."

        # Strict medical keyword filter - only allow medical/health-related questions
        lower_msg = user_message.lower().strip()
        medical_keywords = [
            # Symptoms and sensations
            "pain", "ache", "hurt", "sore", "tender", "discomfort", "feeling", "feel", "feels",
            "numb", "tingling", "burning", "stabbing", "throbbing", "sharp", "dull",
            # Body parts and locations (medical context)
            "head", "chest", "stomach", "back", "neck", "shoulder", "arm", "leg", "foot", "hand",
            "heart", "brain", "tumor", "disease", "symptom", "symptoms",
            # Medical terms
            "doctor", "hospital", "medicine", "medication", "drug", "pill", "tablet", "medical", 
            "mri", "scan", "test", "diagnosis", "treatment", "therapy", "clinic", "patient",
            # Health conditions
            "blood", "pressure", "cholesterol", "fever", "cough", "cold", "flu", "infection",
            "health", "healthy", "diet", "exercise", "sleep", "fatigue", "tired", "illness",
            # Emotional and physical feelings (medical context)
            "nausea", "dizzy", "dizziness", "weak", "weakness", "tired", "fatigue", "anxious", 
            "anxiety", "stress", "depressed", "depression", "sad", "worried", "concerned",
            # Medications (common examples)
            "panadol", "paracetamol", "ibuprofen", "aspirin", "tylenol", "advil", "motrin",
            "antibiotic", "antidepressant", "painkiller", "analgesic", "anti-inflammatory",
            # Additional medical terms
            "wound", "injury", "fracture", "sprain", "inflammation", "swelling", "rash",
            "allergy", "allergic", "breathing", "respiratory", "cardiac", "neurological"
        ]

        # Strict check - refuse non-medical questions
        if len(lower_msg) > 2 and not any(keyword in lower_msg for keyword in medical_keywords):
            return (
                "**📋 Analysis**\n"
                "I'm Dr. MDDS, a medical AI assistant specialized in health and medical questions.\n\n"
                "**💡 Key Information**\n"
                "• I can only answer medical, health, and wellness-related questions\n"
                "• I can help with symptoms, pain, medications, diseases, treatments, and health concerns\n"
                "• I cannot answer questions about non-medical topics (technology, entertainment, general knowledge, etc.)\n\n"
                "**⚡ Next Steps**\n"
                "Please ask me about medical symptoms, health concerns, medications, pain, feelings related to health, or questions about your analysis results.\n\n"
                "**⚠️ Important**\n"
                "For medical questions, I'm here to help! Please rephrase your question with a medical or health focus."
            )
        
        system_prompt = self._build_system_prompt()
        medical_context = self._build_user_medical_context(user_id)
        client = self._get_client()
        
        # Compose messages for Groq chat completion
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "system",
                "content": (
                    "PATIENT'S LATEST ANALYSIS RESULTS (ALWAYS REFERENCE THESE WHEN RELEVANT):\n"
                    f"{medical_context}\n\n"
                    "IMPORTANT: When the patient asks about pain, feelings, symptoms, or medications, "
                    "check if their question relates to their heart or brain analysis results above. "
                    "If relevant, incorporate this information into your response to provide personalized medical guidance."
                ),
            },
            {
                "role": "user",
                "content": f"{user_message}\n\nPlease provide a structured response using the required format with clear sections: 📋 Analysis, 💡 Key Information (with bullet points), ⚡ Next Steps, and ⚠️ Important.",
            },
        ]

        try:
            completion = client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature = 0.4,  # Slightly higher for more natural medical explanations
                max_tokens = 300,  # Reduced for concise, focused responses
            )
        except Exception as e:
            # If Groq call fails, return a graceful message
            print(f"[ERROR] Groq API call failed: {e}")
            return (
                "I’m sorry, but I’m having trouble contacting the AI model right now. "
                "Please try again later."
            )

        # Extract the assistant's reply text
        try:
            reply = completion.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Unexpected Groq response format: {e}")
            return (
                "I received an unexpected response format from the AI model. "
                "Please try again later."
            )

        return reply

# Singleton instance to be imported in routes
chatbot_service = ChatbotService()