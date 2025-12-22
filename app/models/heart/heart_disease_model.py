from typing import Dict, Tuple, Any, List
from pathlib import Path
import numpy as np
import joblib
from app.models.base_model import BaseDiseaseModel

class HeartDiseaseModel(BaseDiseaseModel):    # Wrapper for the Heart Disease prediction model
    def __init__(self, model_path: str | None = None) -> None:
        # Preserve existing default path behavior
        if model_path is None:
            resolved_path = "app/data/saved_models/heart_model.pkl"
        else:
            resolved_path = model_path

        # Initialize common base attributes (_model_path, _loaded_model)
        super().__init__(resolved_path)

        # Keep original public attributes for backward compatibility
        self.model_path: str = str(self._model_path)
        self.loaded_model: Any | None = self._loaded_model

        self.feature_names: List[str] = []

    def load_model(self) -> None:   # Load the RandomForest model + feature names
        if self._loaded_model is not None:
            return  # already loaded

        bundle_path = Path(self._model_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Heart model file not found at: {self.model_path}. "
                f"Make sure you ran the training script and saved the model."
            )

        bundle = joblib.load(bundle_path)
        self._loaded_model = bundle["model"]
        # Keep public alias in sync for any external code that might use it
        self.loaded_model = self._loaded_model
        self.feature_names = bundle["feature_names"]

        if not self.feature_names:
            raise ValueError("Loaded heart model has empty feature_names list.")

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        # Returns -> (risk_label, probability_of_disease)
        self.load_model()

        # Build feature vector in correct order
        row_values: list[float] = []
        for name in self.feature_names:
            value = features.get(name, 0.0)  # default 0 for missing fields
            try:
                row_values.append(float(value))
            except (TypeError, ValueError):
                row_values.append(0.0)

        X = np.array([row_values], dtype = float)

        # Predict probability of each class
        assert self._loaded_model is not None
        proba = self._loaded_model.predict_proba(X)[0]  # shape: (n_classes)

        # We assume class "1" = has disease; figure out which index that is
        classes = list(self.loaded_model.classes_)
        if 1 in classes:
            idx_disease = classes.index(1)
        else:
            # Fallback: assume last class is "disease"
            idx_disease = len(classes) - 1

        prob_disease = float(proba[idx_disease])

        # Map probability to simple risk label
        if prob_disease >= 0.7:
            label = "High"
        elif prob_disease >= 0.4:
            label = "Medium"
        else:
            label = "Low"

        return label, prob_disease
