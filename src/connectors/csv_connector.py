from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.connectors.base import DataConnector
from src.data.loader import CsvSource, load_csv
from src.data.schema import build_schema_summary


@dataclass
class CsvConnector(DataConnector):
    """Adapter that exposes the existing safe CSV loader through DataConnector."""

    source: CsvSource
    _dataframe: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def load(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = load_csv(self.source)
        return self._dataframe

    def get_schema(self) -> dict[str, Any]:
        return build_schema_summary(self.load())
