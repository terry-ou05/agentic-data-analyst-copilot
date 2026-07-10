from io import StringIO
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


CsvSource = Union[str, Path, BinaryIO]
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


class CsvLoadError(ValueError):
    """Raised when a CSV cannot be read safely and predictably."""


def _read_source_bytes(source: CsvSource) -> bytes:
    if isinstance(source, (str, Path)):
        try:
            return Path(source).read_bytes()
        except OSError as exc:
            raise CsvLoadError("The CSV file could not be read.") from exc

    try:
        if hasattr(source, "getvalue"):
            raw_data = source.getvalue()
        else:
            source.seek(0)
            raw_data = source.read()
    except (AttributeError, OSError) as exc:
        raise CsvLoadError("The uploaded CSV could not be read.") from exc

    if isinstance(raw_data, memoryview):
        raw_data = raw_data.tobytes()

    if not isinstance(raw_data, bytes):
        raise CsvLoadError("The uploaded CSV must contain binary file data.")

    return raw_data


def _looks_like_binary(raw_data: bytes) -> bool:
    if b"\x00" in raw_data:
        return True

    control_bytes = sum(
        byte < 32 and byte not in (9, 10, 13)
        for byte in raw_data
    )
    return bool(raw_data) and control_bytes / len(raw_data) > 0.05


def load_csv(source: CsvSource) -> pd.DataFrame:
    """Load a CSV file from a local path or Streamlit uploaded file."""
    raw_data = _read_source_bytes(source)

    if not raw_data or not raw_data.strip():
        raise CsvLoadError("The CSV file is empty or contains only whitespace.")

    if _looks_like_binary(raw_data):
        raise CsvLoadError("The uploaded file does not appear to be a text CSV.")

    decoded_text = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            decoded_text = raw_data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise CsvLoadError(
            "The CSV encoding is not supported. Use UTF-8 or GB18030."
        )

    if not decoded_text.strip():
        raise CsvLoadError("The CSV file is empty or contains only whitespace.")

    try:
        return pd.read_csv(StringIO(decoded_text))
    except EmptyDataError as exc:
        raise CsvLoadError("The CSV file does not contain any parsable columns.") from exc
    except ParserError as exc:
        raise CsvLoadError("The CSV format is invalid or malformed.") from exc
