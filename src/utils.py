"""
src/utils.py
Validaciones y utilidades generales.
"""
import pandas as pd


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Lanza AssertionError si faltan columnas requeridas en el DataFrame.

    >>> import pandas as pd
    >>> assert_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"])  # OK
    >>> assert_columns(pd.DataFrame({"a": [1]}), ["a", "b"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    AssertionError: Columnas faltantes: {'b'}
    """
    missing = set(required) - set(df.columns)
    assert not missing, f"Columnas faltantes: {missing}"


def assert_no_nulls(df: pd.DataFrame, cols: list[str] | None = None) -> None:
    """Lanza AssertionError si hay nulos en las columnas indicadas (o en todo el df)."""
    check = df[cols] if cols else df
    nulls = check.isnull().sum().sum()
    assert nulls == 0, f"Se encontraron {nulls} nulos tras la limpieza"


def assert_target_binary(df: pd.DataFrame, target: str = "num") -> None:
    """Comprueba que la columna target solo tenga valores 0 y 1."""
    vals = set(df[target].unique())
    assert vals <= {0, 1}, f"Target '{target}' tiene valores inesperados: {vals}"


def report_shape(df: pd.DataFrame, label: str = "") -> None:
    """Imprime el shape del DataFrame con una etiqueta descriptiva."""
    tag = f"[{label}] " if label else ""
    print(f"{tag}{df.shape[0]} filas x {df.shape[1]} columnas")
