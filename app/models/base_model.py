from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDiseaseModel(ABC):
    """Abstract base class for disease prediction models.

    Provides a common interface so different disease models can be treated
    polymorphically (e.g., in a list[BaseDiseaseModel]) while keeping each
    model's internal details independent.
    """

    def __init__(self, model_path: str | Path) -> None:
        # Protected attributes used by concrete models
        self._model_path: Path = Path(model_path)
        self._loaded_model: Any | None = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the underlying ML model into memory."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run a prediction using the loaded model."""
