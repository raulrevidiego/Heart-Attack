"""
src/io.py
Funciones de carga y exportación de datos.
"""
import pandas as pd
from pathlib import Path


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Carga un CSV y elimina espacios en los nombres de columna."""
    df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.str.strip()
    print(f"[io] Cargado: {path}  →  {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def export_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Exporta un DataFrame a CSV, creando la carpeta si no existe."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, **kwargs)
    print(f"[io] Exportado: {out}  →  {df.shape[0]} filas x {df.shape[1]} columnas")
