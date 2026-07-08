import pandas as pd


def build_schema_summary(dataframe: pd.DataFrame) -> dict:
    """Create the dataset summary shown in the Streamlit app."""
    row_count = int(dataframe.shape[0])
    missing_values = dataframe.isna().sum()
    if row_count:
        missing_percentage = (missing_values / row_count * 100).round(2)
    else:
        missing_percentage = pd.Series(0, index=dataframe.columns)

    schema_table = pd.DataFrame(
        {
            "column name": dataframe.columns,
            "data type": dataframe.dtypes.astype(str).values,
            "missing values": missing_values.values,
            "missing percentage": missing_percentage.values,
        }
    )

    return {
        "number_of_rows": row_count,
        "number_of_columns": int(dataframe.shape[1]),
        "column_names": list(dataframe.columns),
        "total_missing_values": int(missing_values.sum()),
        "schema_table": schema_table,
    }
