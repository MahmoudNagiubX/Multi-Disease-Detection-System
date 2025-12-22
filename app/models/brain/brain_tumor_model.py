from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array   # type: ignore

from app.models.base_model import BaseDiseaseModel


class BrainTumorModel(BaseDiseaseModel):  # Wrapper around the trained 4-class CNN for brain tumor detection

    def __init__(
        self,
        model_path: str | Path | None = None,
        img_size: tuple[int, int] = (128, 128),
    ) -> None:
        # Locate app directory: .../Multi Disease Detection System/app
        app_dir = Path(__file__).resolve().parents[2]

        # Default model path (must match your training script)
        if model_path is None:
            model_path = app_dir / "data" / "saved_models" / "brain_tumor_cnn_multiclass.h5"

        # Initialize common base attributes (_model_path, _loaded_model)
        super().__init__(model_path)

        # Keep original public attribute for backward compatibility
        self.model_path: Path = self._model_path

        self.img_size: tuple[int, int] = img_size

        # Keras model instance used internally by this subclass
        self._model: keras.Model | None = None

        # Make sure this matches class_names from training
        self.class_names: List[str] = [
            "glioma",
            "meningioma",
            "no_tumor",
            "pituitary",
        ]
        
    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Brain tumor model file not found at: {self.model_path}\n"
                "Make sure you have trained the 4-class model and saved it "
                "to this location."
            )

        self._model = keras.models.load_model(self.model_path)  # Load the trained CNN
        print(f"[BrainTumorModel] Loaded model from: {self.model_path}")

    def load_model(self) -> None:
        """Concrete implementation of the abstract load_model interface."""
        self._ensure_model_loaded()

    def _preprocess_image(self, image_path: str | Path) -> np.ndarray:
        """
        Preprocess image for model prediction.
        - Converts image path to Path object
        - Loads and resizes image to model input size (128x128)
        - Converts to RGB format
        - Returns pixel values in [0, 255] range (model has Rescaling layer)
        - Adds batch dimension
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            # Load & resize image to RGB format
            # load_img returns a PIL Image in RGB mode
            img = load_img(
                str(image_path),  # Ensure string path for load_img
                target_size=self.img_size,
                color_mode="rgb",
            )
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {str(e)}")
        
        try:
            # Convert PIL image to numpy array
            # img_to_array returns values in [0, 255] range by default
            img_array = img_to_array(img).astype("float32")
            
            # The model has a Rescaling(1.0/255) layer built-in, so it expects
            # pixel values in [0, 255] range and will normalize internally
            # Do NOT normalize here to avoid double normalization
            
            # Verify values are in expected range
            if img_array.min() < 0 or img_array.max() > 255:
                raise ValueError(
                    f"Image pixel values out of range [0, 255]: "
                    f"min={img_array.min()}, max={img_array.max()}"
                )
            
            # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Verify shape is correct: should be (1, 128, 128, 3)
            expected_shape = (1,) + self.img_size + (3,)
            if img_array.shape != expected_shape:
                raise ValueError(
                    f"Unexpected image shape: {img_array.shape}. Expected: {expected_shape}"
                )
            
            return img_array
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        self._ensure_model_loaded()

        assert self._model is not None  # for type checkers

        # Preprocess image
        x = self._preprocess_image(image_path)

        # Model returns shape (1, num_classes); we take [0]
        preds: np.ndarray = self._model.predict(x, verbose=0)[0]

        # Convert to python floats
        preds = preds.astype("float64")
        pred_index = int(np.argmax(preds))
        probability = float(preds[pred_index])

        # Safety check: class_names length should match num_classes
        if pred_index < len(self.class_names):
            predicted_class = self.class_names[pred_index]
        else:
            predicted_class = f"class_{pred_index}"

        probabilities_dict: Dict[str, float] = {}
        for i, p in enumerate(preds):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            probabilities_dict[name] = float(p)

        return {
            "predicted_class": predicted_class,
            "predicted_index": pred_index,
            "probability": probability,
            "probabilities": probabilities_dict,
        }
