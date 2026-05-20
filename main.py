"""
main.py
Entrypoint reproducible del proyecto Heart Attack Risk.

Uso:
    python main.py
"""
from pathlib import Path

from src.io       import load_csv, export_csv
from src.cleaning import clean
from src.features import build_features
from src.utils    import assert_no_nulls, assert_target_binary, report_shape
from src.viz      import (setup_style, plot_class_balance,
                          plot_continuous_distributions,
                          plot_categorical_counts,
                          plot_correlation_heatmap,
                          plot_risk_score_distribution,
                          plot_age_group_risk)

RAW_PATH       = Path("data/raw/heart_attack.csv")
PROCESSED_PATH = Path("data/processed/heart_attack_clean.csv")


def main() -> None:
    setup_style()

    # 1. Cargar
    print("\n── 1. CARGA ─────────────────────────────────────")
    df_raw = load_csv(RAW_PATH)
    report_shape(df_raw, "raw")

    # 2. Limpiar (devuelve df + quality report + imputation impact)
    print("\n── 2. LIMPIEZA ──────────────────────────────────")
    df_clean, quality_report, imputation_impact = clean(df_raw)
    assert_no_nulls(df_clean)
    assert_target_binary(df_clean, target="num")

    print("\n=== QUALITY REPORT ===")
    print(quality_report.to_string())
    print("\n=== IMPUTATION IMPACT ===")
    print(imputation_impact.to_string())

    # 3. Feature engineering
    print("\n── 3. FEATURES ──────────────────────────────────")
    df_feat = build_features(df_clean)
    report_shape(df_feat, "features")

    # 4. Exportar
    print("\n── 4. EXPORTAR ──────────────────────────────────")
    export_csv(df_feat, PROCESSED_PATH)

    # 5. Visualizar
    print("\n── 5. VISUALIZACIONES ───────────────────────────")
    plot_class_balance(df_clean)
    plot_continuous_distributions(df_clean)
    plot_categorical_counts(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_risk_score_distribution(df_feat)
    plot_age_group_risk(df_feat)

    print("\n✅ Pipeline completado correctamente.")


if __name__ == "__main__":
    main()
