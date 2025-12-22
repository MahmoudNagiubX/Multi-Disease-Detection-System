from __future__ import annotations
from abc import ABC
from app.core.managers.database_manager import DatabaseManager


class BaseService(ABC):
    """Abstract base class for services that work with the database.
    Provides a shared `_db` attribute so services can be treated uniformly,
    while keeping each service's public API unchanged.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db: DatabaseManager = db