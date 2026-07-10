import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd


def build_schema_signature(schema_summary: Mapping[str, Any]) -> str:
    """Build a stable signature for the schema and row count bound to a plan."""
    column_names = schema_summary.get("column_names")
    number_of_rows = schema_summary.get("number_of_rows")
    schema_table = schema_summary.get("schema_table")

    if not isinstance(column_names, list) or any(
        not isinstance(column, str) for column in column_names
    ):
        raise ValueError("Schema column_names must be a list of strings.")
    if not isinstance(number_of_rows, int) or isinstance(number_of_rows, bool):
        raise ValueError("Schema number_of_rows must be an integer.")
    if not isinstance(schema_table, pd.DataFrame) or "data type" not in schema_table:
        raise ValueError("Schema table must contain data type information.")

    column_types = schema_table["data type"].astype(str).tolist()
    if len(column_types) != len(column_names):
        raise ValueError("Schema column names and data types must have equal length.")

    signature_payload = {
        "column_names": column_names,
        "column_types": column_types,
        "number_of_rows": number_of_rows,
    }
    serialized = json.dumps(
        signature_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
