import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.connectors.base import DataConnector
from src.data.schema import build_schema_summary


class SQLiteConnectorError(ValueError):
    """Raised when a SQLite source cannot be safely listed or loaded."""


@dataclass
class SQLiteConnector(DataConnector):
    """Read-only SQLite adapter that loads one metadata-verified table."""

    database_path: str | Path
    table_name: str | None = None
    _dataframe: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def _resolve_database_path(self) -> Path:
        try:
            path = Path(self.database_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SQLiteConnectorError("SQLite database file was not found.") from exc
        if not path.is_file():
            raise SQLiteConnectorError("SQLite database file was not found.")
        return path

    def _connect_readonly(self) -> sqlite3.Connection:
        database_uri = f"{self._resolve_database_path().as_uri()}?mode=ro"
        try:
            return sqlite3.connect(database_uri, uri=True)
        except sqlite3.Error as exc:
            raise SQLiteConnectorError("SQLite database could not be opened.") from exc

    def list_tables(self) -> list[str]:
        connection = self._connect_readonly()
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        except sqlite3.Error as exc:
            raise SQLiteConnectorError("SQLite table metadata could not be read.") from exc
        finally:
            connection.close()
        return [str(row[0]) for row in rows]

    def _selected_table(self) -> str:
        if not isinstance(self.table_name, str) or not self.table_name.strip():
            raise SQLiteConnectorError("Select a SQLite table before loading data.")
        table_name = self.table_name.strip()
        if table_name not in self.list_tables():
            raise SQLiteConnectorError("Selected SQLite table is unavailable.")
        return table_name

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def load(self) -> pd.DataFrame:
        if self._dataframe is not None:
            return self._dataframe

        table_name = self._selected_table()
        connection = self._connect_readonly()
        try:
            self._dataframe = pd.read_sql_query(
                f"SELECT * FROM {self._quote_identifier(table_name)}",
                connection,
            )
        except (sqlite3.Error, ValueError) as exc:
            raise SQLiteConnectorError("Selected SQLite table could not be loaded.") from exc
        finally:
            connection.close()
        return self._dataframe

    def get_schema(self) -> dict[str, Any]:
        return build_schema_summary(self.load())
