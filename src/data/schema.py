import pandas as pd


def build_schema_summary(dataframe: pd.DataFrame) -> dict:
    """Create the basic dataset summary shown in the V1 app."""
    return {
        "number_of_rows": int(dataframe.shape[0]),
        "number_of_columns": int(dataframe.shape[1]),
        "column_names": list(dataframe.columns),
        "data_types": dataframe.dtypes.astype(str).reset_index().rename(
            columns={"index": "column", 0: "dtype"}
        ),
        "missing_values": dataframe.isna().sum().reset_index().rename(
            columns={"index": "column", 0: "missing_values"}
        ),
    }
