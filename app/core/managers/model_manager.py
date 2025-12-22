from typing import Optional
from app.models.heart.heart_disease_model import HeartDiseaseModel
from app.models.brain.brain_tumor_model import BrainTumorModel

class ModelManager: # Manages ML/DL model instances
    def __init__(self) -> None:
        self._heart_model: Optional[HeartDiseaseModel] = None
        self._brain_model: Optional[BrainTumorModel] = None
        self._heart_model_error: Optional[str] = None
        self._brain_model_error: Optional[str] = None
        
    def get_heart_model(self) -> HeartDiseaseModel: #  Return a loaded HeartDiseaseModel instance
        if self._heart_model is None and self._heart_model_error is None:
            try:
                # In the future we may pass a real model_path here.
                self._heart_model = HeartDiseaseModel(
                    model_path="app/data/saved_models/heart_model.pkl"
                )
                self._heart_model.load_model()
            except FileNotFoundError as e:
                self._heart_model_error = f"Heart disease model file not found. Please ensure the model is trained and saved."
                print(f"[ERROR] ModelManager: {self._heart_model_error}: {e}")
                raise RuntimeError(self._heart_model_error)
            except Exception as e:
                self._heart_model_error = f"Failed to load heart disease model: {str(e)}"
                print(f"[ERROR] ModelManager: {self._heart_model_error}")
                raise RuntimeError(self._heart_model_error)
        
        if self._heart_model is None:
            raise RuntimeError(self._heart_model_error or "Heart model failed to load.")
        
        return self._heart_model

    def get_brain_model(self) -> BrainTumorModel:   # Return a loaded BrainTumorModel instance
        if self._brain_model is None and self._brain_model_error is None:
            try:
                # Model will use default path or can be overridden
                # The model loads lazily when predict() is first called
                self._brain_model = BrainTumorModel()
            except Exception as e:
                self._brain_model_error = f"Failed to initialize brain tumor model: {str(e)}"
                print(f"[ERROR] ModelManager: {self._brain_model_error}")
                raise RuntimeError(self._brain_model_error)
        
        if self._brain_model is None:
            raise RuntimeError(self._brain_model_error or "Brain model failed to load.")
        
        return self._brain_model

# Global instance used by services
model_manager = ModelManager()