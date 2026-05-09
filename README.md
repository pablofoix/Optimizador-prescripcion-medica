# 💊 Optimizador de Prescripción Médica

Motor de clasificación basado en redes neuronales (MLP) para optimizar la asignación de tratamientos farmacéuticos según el perfil clínico del paciente.

---

## 📌 Descripción

Este proyecto aborda un problema de clasificación multiclase: dado el perfil de un paciente (edad, sexo, presión arterial, colesterol y relación sodio/potasio), predecir cuál de los cinco tratamientos farmacéuticos disponibles es el más adecuado.

Se entrenó un **MLPClassifier** (red neuronal multicapa) con búsqueda de hiperparámetros mediante **RandomizedSearchCV**, logrando alta precisión en el conjunto de validación.

---

## 🗂️ Estructura del proyecto

```
Optimizador-Prescripcion-Medica/
├── data/
│   └── drugs_classif.csv           # Dataset de pacientes
├── notebooks/
│   └── eda.ipynb          # Análisis exploratorio de datos
├── src/
│   └── train.py           # Pipeline de entrenamiento y evaluación
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

El dataset contiene **200 registros** de pacientes con las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `Age` | Numérica | Edad del paciente |
| `Sex` | Categórica | Sexo (M / F) |
| `BP` | Categórica | Presión arterial (LOW / NORMAL / HIGH) |
| `Cholesterol` | Categórica | Nivel de colesterol (NORMAL / HIGH) |
| `Na_to_K` | Numérica | Relación sodio/potasio en sangre |
| `Drug` | Target | Tratamiento asignado (drugA / B / C / X / Y) |

---

## ⚙️ Pipeline

```
Carga de datos
    └── Encoding de variables categóricas
    └── Filtro de outliers (método Hampel) sobre Na_to_K
        └── Train/Test split (80/20)
            └── Escalado (RobustScaler)
                └── Búsqueda de hiperparámetros (RandomizedSearchCV)
                    └── Entrenamiento del modelo final (MLPClassifier)
                        └── Evaluación (accuracy, reporte de clasificación, matriz de confusión)
```

---

## 🧠 Modelo

**MLPClassifier** — Red neuronal multicapa (scikit-learn)

| Parámetro | Valor | Justificación |
|---|---|---|
| `hidden_layer_sizes` | (5, 10) | Dos capas que captan relaciones no lineales sin sobrecomplicar la arquitectura |
| `solver` | adam | Eficiente para datasets de tamaño mediano |
| `learning_rate_init` | 0.01 | Convergencia estable sin oscilaciones |
| `alpha` | 1 | Regularización L2 moderada para controlar el sobreajuste |
| `max_iter` | 1000 | Iteraciones suficientes para la convergencia |

---

## 📈 Resultados

| Métrica | Score |
|---|---|
| Train Accuracy | ~0.97 |
| Test Accuracy | ~0.95 |
| Null Accuracy | ~0.47 |

El modelo supera ampliamente el azar y no presenta sobreajuste significativo.

---

## 🚀 Cómo ejecutar

**1. Clonar el repositorio**
```bash
git clone https://github.com/pablofoix/Optimizador-Prescripcion-Medica.git
cd Optimizador-Prescripcion-Medica
```

**2. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**3. Ejecutar el pipeline de entrenamiento**
```bash
python src/train.py
```

**4. Explorar el análisis exploratorio**

Abrir `notebooks/eda.ipynb` en Jupyter Notebook o JupyterLab.

---

## 🛠️ Tech Stack

- Python 3.10+
- pandas / numpy
- scikit-learn
- matplotlib / seaborn

---

## 👤 Autor

**Pablo Foix** — Data Analyst Jr

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pablofoix)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/pablofoix)
