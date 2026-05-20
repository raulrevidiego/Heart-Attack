# Predicción de Riesgo de Ataque Cardíaco

## Objetivo
Explorar qué variables clínicas y demográficas se asocian con mayor riesgo de ataque cardíaco, y construir modelos de clasificación que ayuden a identificar pacientes de alto riesgo.

## Dataset
- **Nombre:** Heart Attack Analysis & Prediction Dataset
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
- **Archivo local:** `data/raw/heart_attack.csv`
- **Filas:** ~303 pacientes | **Target:** `num` (0 = bajo riesgo, 1 = alto riesgo)

### Variables principales
| Variable | Descripción |
|---|---|
| `age` | Edad del paciente |
| `sex` | Género (1=hombre, 0=mujer) |
| `cp` | Tipo de dolor en pecho (0-3) |
| `trestbps` | Presión arterial en reposo (mm Hg) |
| `chol` | Colesterol (mg/dl) |
| `fbs` | Azúcar en sangre en ayunas (1 = >120 mg/dl) |
| `restecg` | Resultados electrocardiográficos (0-2) |
| `thalach` | Frecuencia cardíaca máxima alcanzada |
| `exang` | Angina inducida por ejercicio (1=sí) |
| `oldpeak` | Depresión segmento ST por ejercicio |
| `num` | **Target**: 0=bajo riesgo, 1=alto riesgo |

## Preguntas de análisis
1. ¿Están balanceadas las clases de riesgo?
2. ¿Qué variables continuas separan mejor a los grupos?
3. ¿Qué variables categóricas tienen mayor impacto en el riesgo?
4. ¿Qué modelo clasifica mejor con este dataset pequeño?

## Pipeline

```
data/raw/heart_attack.csv
    ↓ load_csv()         (src/io.py)
    ↓ clean()            (src/cleaning.py)   — reemplaza '?', imputa KNN, elimina cols >40% nulos
    ↓ build_features()   (src/features.py)   — age_group, risk_score, normalización
    ↓ export_csv()       (src/io.py)         → data/processed/heart_attack_clean.csv
    ↓ plot_*()           (src/viz.py)        — 6 gráficos
```

Para ejecutar el pipeline completo:
```bash
python main.py
```

Para exploración interactiva:
```bash
cd notebooks && jupyter notebook eda.ipynb
```

## Estructura del proyecto
```
heart_attack_project/
├── main.py                  # Entrypoint reproducible
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/
│   │   └── heart_attack.csv     ← pon tu CSV aquí
│   └── processed/               ← generado por el pipeline (en .gitignore)
├── notebooks/
│   └── eda.ipynb
└── src/
    ├── __init__.py
    ├── io.py                # load_csv, export_csv
    ├── cleaning.py          # clean, impute_knn, drop_high_null_cols
    ├── features.py          # build_features, add_age_group, add_risk_score
    ├── viz.py               # plot_class_balance, plot_continuous_distributions, …
    └── utils.py             # assert_columns, assert_no_nulls, assert_target_binary
```

## Hallazgos principales

1. **`cp` y `oldpeak` son los mejores predictores** — el mapa de correlación y los KDE plots muestran que pacientes con `oldpeak ≈ 0` se concentran casi exclusivamente en bajo riesgo.

2. **k-NN (k=25) supera a Random Forest** — con ~294 filas los modelos complejos sobreajustan. El k-NN alcanza ROC-AUC ≈ 0.90 y Accuracy ≈ 88%.

3. **La `risk_score` heurística funciona como regla de triaje** — combinando cp, exang y oldpeak en un score 0-3, los pacientes con score=3 son mayoritariamente de alto riesgo.
