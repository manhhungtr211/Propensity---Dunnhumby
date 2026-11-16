"""CSV parsing preserves declared column semantics and missing tokens."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from propensity.io.csv import read_csv
from propensity.overview.missing import missing_mask
from propensity.schema import DatasetSpec, DatasetSpecError


@pytest.mark.parametrize(
    "rows",
    ["1,0,001\n,1,010\n", "2,A,B\n3,0,002\n"],
)
def test_read_csv_when_codes_vary_preserves_declared_dtypes(
    tmp_path: Path, rows: str
) -> None:
    # Given
    path = tmp_path / "causal.csv"
    path.write_text("PRODUCT_ID,display,mailer\n" + rows, encoding="utf-8")
    spec = DatasetSpec(
        "causal", identifier_cols=("PRODUCT_ID",), string_cols=("display", "mailer")
    )
    # When
    frame = read_csv(path, spec=spec)
    # Then
    assert frame["PRODUCT_ID"].dtype == pd.Int64Dtype()
    assert isinstance(frame["display"].dtype, pd.StringDtype)
    assert isinstance(frame["mailer"].dtype, pd.StringDtype)
    assert frame["display"].tolist() == [row.split(",")[1] for row in rows.splitlines()]
    assert frame["mailer"].tolist() == [row.split(",")[2] for row in rows.splitlines()]


def test_read_csv_when_custom_tokens_extend_pandas_defaults(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "tokens.csv"
    path.write_text(
        'id,code\n1,MISSING\n2,NA\n3,\n4," \t"\n5,0\n6,001\n7," x "\n',
        encoding="utf-8",
    )
    spec = DatasetSpec("tokens", string_cols=("code",), na_tokens=("MISSING",))
    # When
    frame = read_csv(path, spec=spec)
    # Then
    assert_series_equal(
        frame["code"],
        pd.Series(
            [pd.NA, pd.NA, pd.NA, " \t", "0", "001", " x "], dtype="string", name="code"
        ),
    )
    assert missing_mask(frame["code"]).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_read_csv_when_no_spec_uses_configured_default_tokens(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "tokens.csv"
    path.write_text("id,code\n1,?\n2,na\n3,0\n", encoding="utf-8")
    # When
    frame = read_csv(path)
    # Then
    assert frame["code"].isna().tolist() == [True, True, False]
    assert frame.loc[2, "code"] == 0


def test_read_csv_when_datetime_and_measure_columns_exist(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "dated.csv"
    path.write_text(
        "id,date,amount\n1,2024-01-01,1.5\n,2024-01-02,2.5\n", encoding="utf-8"
    )
    spec = DatasetSpec("dated", identifier_cols=("id",), datetime_cols=("date",))
    # When
    frame = read_csv(path, spec=spec)
    # Then
    assert frame["id"].dtype == pd.Int64Dtype()
    assert frame["id"].isna().tolist() == [False, True]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert frame["amount"].tolist() == [1.5, 2.5]


@pytest.mark.parametrize(
    "spec",
    [
        DatasetSpec("required", required_cols=("absent",)),
        DatasetSpec("identifier", identifier_cols=("absent",)),
        DatasetSpec("datetime", datetime_cols=("absent",)),
    ],
)
def test_read_csv_when_required_column_is_absent_raises_typed_error(
    tmp_path: Path, spec: DatasetSpec
) -> None:
    # Given
    path = tmp_path / "missing.csv"
    path.write_text("present\n1\n", encoding="utf-8")
    # When / Then
    with pytest.raises(DatasetSpecError) as error:
        read_csv(path, spec=spec)
    assert spec.name in error.value.reason
    assert "absent" in error.value.reason


def test_read_csv_when_csv_is_malformed_raises_parser_error(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "broken.csv"
    path.write_text('id,code\n1,"unterminated\n', encoding="utf-8")
    # When / Then
    with pytest.raises(pd.errors.ParserError):
        read_csv(path)


def test_read_csv_when_file_is_absent_raises_file_not_found(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "absent.csv"
    # When / Then
    with pytest.raises(FileNotFoundError) as error:
        read_csv(path)
    assert error.value.args == (path,)
