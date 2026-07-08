from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


CsvSource = Union[str, Path, BinaryIO]


def load_csv(source: CsvSource) -> pd.DataFrame:
    """Load a CSV file from a local path or Streamlit uploaded file."""
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        return pd.read_csv(source, encoding="utf-8-sig")
