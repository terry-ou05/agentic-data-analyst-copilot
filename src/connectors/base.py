from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataConnector(ABC):
    """Minimal source boundary that supplies a dataframe and its schema."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the authorized source into a pandas DataFrame."""

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return the standard schema summary for the loaded dataframe."""
