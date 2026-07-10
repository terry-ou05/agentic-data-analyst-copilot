import sqlite3
from pathlib import Path

import pandas as pd


SYNTHETIC_TABLE_NAME = "orders"


def generate_synthetic_business_dataframe() -> pd.DataFrame:
    """Create a deterministic business dataset for offline workflow evaluation."""
    category_specs = (
        ("Laptop", "Computer", 900, 14),
        ("Monitor", "Display", 250, 9),
        ("Keyboard", "Accessory", 80, 30),
    )
    region_quantity_adjustments = {"North": 4, "South": 2, "West": 0}
    records: list[dict[str, object]] = []

    for day_index in range(1, 6):
        date = f"2026-02-{day_index:02d}"
        for product, category, unit_price, base_quantity in category_specs:
            for region, adjustment in region_quantity_adjustments.items():
                quantity = base_quantity + adjustment + day_index
                channel = "Online" if region != "West" else "Retail"
                revenue = quantity * unit_price
                records.append(
                    {
                        "date": date,
                        "product": product,
                        "category": category,
                        "region": region,
                        "channel": channel,
                        "quantity": quantity,
                        "revenue": revenue,
                    }
                )

    return pd.DataFrame.from_records(records)


def create_synthetic_sqlite_database(
    database_path: str | Path,
    table_name: str = SYNTHETIC_TABLE_NAME,
) -> Path:
    """Store a deterministic synthetic dataset in a new SQLite database file."""
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("table_name must be a non-empty string.")

    target_path = Path(database_path).expanduser()
    if target_path.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation database.")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = generate_synthetic_business_dataframe()
    try:
        with sqlite3.connect(target_path) as connection:
            dataframe.to_sql(table_name.strip(), connection, if_exists="fail", index=False)
    except Exception:
        if target_path.exists():
            target_path.unlink()
        raise
    return target_path
