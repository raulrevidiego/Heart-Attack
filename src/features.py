"""
src/features.py
Ingeniería de características para el dataset de ataque cardíaco.
"""
import pandas as pd


def add_age_group(df: pd.DataFrame, col: str = "age") -> pd.DataFrame:
    """Añade columna 'age_group' con tramos de edad."""
    bins   = [0, 40, 50, 60, 70, 120]
    labels = ["<40", "40-50", "50-60", "60-70", ">70"]
    df = df.copy()
    df["age_group"] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Puntuación de riesgo heurística (0-3) basada en las tres variables
    con mayor correlación con 'num': cp, exang, oldpeak.
    """
    df = df.copy()
    df["risk_score"] = (
        (df["cp"] >= 2).astype(int)
        + (df["exang"] == 1).astype(int)
        + (df["oldpeak"] > 1.5).astype(int)
    )
    return df


def add_thalach_reserve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frecuencia cardíaca de reserva: diferencia entre la máxima teórica (220-edad)
    y la alcanzada. Valores bajos indican menor capacidad cardíaca.
    """
    df = df.copy()
    df["thalach_reserve"] = (220 - df["age"]) - df["thalach"]
    return df


def add_chol_age_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio colesterol/edad: el colesterol alto es más preocupante en jóvenes.
    """
    df = df.copy()
    df["chol_age_ratio"] = (df["chol"] / df["age"]).round(2)
    return df


def add_high_bp_flag(df: pd.DataFrame, threshold: float = 140.0) -> pd.DataFrame:
    """Flag binario de hipertensión (trestbps >= 140 mm Hg)."""
    col = "trestbps" if "trestbps" in df.columns else "trtbps"
    df = df.copy()
    if col in df.columns:
        df["high_bp"] = (df[col] >= threshold).astype(int)
    return df


def normalize_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Normalización min-max de las columnas indicadas (genera nuevas columnas _norm)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de features en orden:
      1. Tramos de edad
      2. Puntuación de riesgo heurística (cp + exang + oldpeak)
      3. Frecuencia cardíaca de reserva
      4. Ratio colesterol/edad
      5. Flag de hipertensión
      6. Normalización min-max de continuas clave
    """
    df = add_age_group(df)
    df = add_risk_score(df)
    df = add_thalach_reserve(df)
    df = add_chol_age_ratio(df)
    df = add_high_bp_flag(df)
    df = normalize_columns(df, ["age", "chol", "oldpeak", "thalach"])
    print(f"[features] Columnas tras feature engineering: {list(df.columns)}")
    return df
