from io import BytesIO

import pytest

from src.data.loader import CsvLoadError, load_csv


UTF8_CSV = "地区,收入\n华南,10\n"


@pytest.mark.parametrize(
    "payload",
    [
        UTF8_CSV.encode("utf-8"),
        UTF8_CSV.encode("utf-8-sig"),
        UTF8_CSV.encode("gbk"),
        UTF8_CSV.encode("gb18030"),
    ],
)
def test_supported_encodings_load(payload) -> None:
    dataframe = load_csv(BytesIO(payload))

    assert list(dataframe.columns) == ["地区", "收入"]
    assert dataframe.shape == (1, 2)


@pytest.mark.parametrize("payload", [b"", b"   \r\n\t"])
def test_empty_or_whitespace_file_fails(payload) -> None:
    with pytest.raises(CsvLoadError, match="empty or contains only whitespace"):
        load_csv(BytesIO(payload))


def test_malformed_csv_fails() -> None:
    with pytest.raises(CsvLoadError, match="invalid or malformed"):
        load_csv(BytesIO(b'category,revenue\n"unterminated,10\n'))


def test_invalid_binary_content_fails() -> None:
    with pytest.raises(CsvLoadError, match="does not appear to be a text CSV"):
        load_csv(BytesIO(b"PK\x03\x04\x00\x00\x00\x00binary"))


def test_header_only_csv_is_allowed() -> None:
    dataframe = load_csv(BytesIO(b"category,revenue\n"))

    assert list(dataframe.columns) == ["category", "revenue"]
    assert dataframe.empty


def test_header_only_summary_blocks_llm_ui(monkeypatch) -> None:
    from app import streamlit_app

    class FakeStreamlit:
        def __init__(self):
            self.warnings = []

        def subheader(self, message):
            pass

        def write(self, message):
            pass

        def info(self, message):
            pass

        def warning(self, message):
            self.warnings.append(message)

        def text_area(self, *args, **kwargs):
            raise AssertionError("The question input must not render for a zero-row CSV")

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)

    streamlit_app.render_analysis_planner({"number_of_rows": 0})

    assert fake_streamlit.warnings
    assert "no data rows" in fake_streamlit.warnings[0]
