"""
src/cleaning.py
Limpieza del dataset de ataque cardíaco.
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.utils import assert_columns

# Columnas requeridas del dataset crudo
REQUIRED_COLS = ["age", "sex", "cp", "chol", "fbs", "restecg", "exang", "oldpeak", "num"]
# Columnas que se convertirán a num, no queresmos object por los '?'
COLS_TO_NUM = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "ca", "thal", "slope"]
# Columnas categóricas que, tras imputar, hay que redondear a entero
COLS_CAT = ["sex", "cp", "fbs", "restecg", "exang", "num"]
# Umbral de nulos a partir del cual se elimina la columna
NULL_THRESHOLD = 0.40


def replace_question_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Sustituye '?' por NaN para que pandas los trate como nulos."""
    return df.replace("?", np.nan)


def cast_to_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Convierte las columnas indicadas a numérico (errors='coerce')."""
    cols = cols or COLS_TO_NUM
    existing = [c for c in cols if c in df.columns]
    df[existing] = df[existing].apply(pd.to_numeric, errors="coerce")
    return df


def drop_high_null_cols(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    """Elimina columnas con más de `threshold` fracción de nulos."""
    porc = df.isnull().mean()
    cols_drop = porc[porc > threshold].index.tolist()
    if cols_drop:
        print(f"[cleaning] Columnas eliminadas (>{threshold*100:.0f}% nulos): {cols_drop}")
    return df.drop(columns=cols_drop)


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Imputa nulos restantes con KNN Imputer."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    arr = imputer.fit_transform(df)
    df_imp = pd.DataFrame(arr, columns=df.columns)

    # Las columnas categóricas quedan con decimales tras imputar → redondeamos
    cat_exist = [c for c in COLS_CAT if c in df_imp.columns]
    df_imp[cat_exist] = df_imp[cat_exist].round().astype(int)
    return df_imp


def quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla resumen de calidad antes/después de la limpieza.
    Muestra nulos, % missing y si la columna fue eliminada.
    """
    nulos_antes = df_before.isnull().sum()
    porc_antes  = (nulos_antes / len(df_before) * 100).round(1)
    cols_eliminadas = set(df_before.columns) - set(df_after.columns)

    filas = []
    for col in df_before.columns:
        eliminada = col in cols_eliminadas
        nulos_despues = 0 if eliminada else df_after[col].isnull().sum()
        filas.append({
            "columna":        col,
            "nulos_antes":    nulos_antes[col],
            "pct_missing":    f"{porc_antes[col]}%",
            "eliminada":      "✗ sí" if eliminada else "✓ no",
            "nulos_después":  "—" if eliminada else nulos_despues,
        })
    return pd.DataFrame(filas).set_index("columna")


def imputation_impact(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """
    Muestra cuántos registros fueron imputados por columna (antes vs después).
    Solo para columnas que sobrevivieron la limpieza.
    """
    cols_comunes = [c for c in df_before.columns if c in df_after.columns]
    filas = []
    for col in cols_comunes:
        n_antes = df_before[col].isnull().sum()
        n_despues = df_after[col].isnull().sum()
        if n_antes > 0:
            filas.append({
                "columna":           col,
                "nulos_antes_KNN":   n_antes,
                "nulos_después_KNN": n_despues,
                "registros_imputados": n_antes - n_despues,
            })
    return pd.DataFrame(filas).set_index("columna")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de limpieza. Devuelve:
      - df_clean   : dataset limpio
      - qr         : quality_report (tabla antes/después)
      - imp_impact : imputation_impact (registros imputados por columna)
    """
    assert_columns(df, REQUIRED_COLS)
    df_raw = df.copy()
    df = replace_question_marks(df)
    df = cast_to_numeric(df)
    df_pre_drop = df.copy()
    df = drop_high_null_cols(df)
    df_pre_knn  = df.copy()
    df = impute_knn(df)

    qr         = quality_report(df_pre_drop, df)
    imp_impact = imputation_impact(df_pre_knn, df)

    print(f"[cleaning] Dataset limpio: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"[cleaning] Nulos restantes: {df.isnull().sum().sum()}")
    return df, qr, imp_impact
