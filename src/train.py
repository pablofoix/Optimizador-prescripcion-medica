# =============================================================================
# Optimizador de Prescripción Médica — Pipeline de Entrenamiento
# Clasificación de tratamientos farmacéuticos mediante redes neuronales (MLP)
# Autor: Pablo Foix
# =============================================================================

import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO
# =============================================================================

def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el dataset y aplica las transformaciones necesarias."""
    df = pd.read_csv(ruta)

    df["Sex"]        = df["Sex"].map({"F": 0, "M": 1})
    df["Cholesterol"] = df["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})
    df["BP"]         = df["BP"].map({"LOW": 1, "NORMAL": 2, "HIGH": 3})
    df["Drug"]       = df["Drug"].map({"drugA": 1, "drugB": 2, "drugC": 3, "DrugY": 4, "drugX": 5})

    return df


# =============================================================================
# 2. DIVISIÓN Y ESCALADO
# =============================================================================

def preparar_datos(df: pd.DataFrame, target: str = "Drug", test_size: float = 0.2):
    """Divide en train/test y aplica RobustScaler."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# =============================================================================
# 3. BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================

def buscar_hiperparametros(X_train_scaled, y_train) -> RandomizedSearchCV:
    """Ejecuta RandomizedSearchCV sobre MLPClassifier."""
    param_distributions = {
        "hidden_layer_sizes": [(5, 5), (5, 10), (10, 10)],
        "alpha": np.logspace(-3, 3, 7),
        "learning_rate_init": [0.001, 0.01, 0.1],
    }

    grid = RandomizedSearchCV(
        estimator=MLPClassifier(solver="adam", max_iter=2000),
        param_distributions=param_distributions,
        n_iter=20,
        scoring="accuracy",
        n_jobs=multiprocessing.cpu_count() - 1,
        cv=4,
        verbose=1,
        random_state=123,
        return_train_score=True
    )

    grid.fit(X=X_train_scaled, y=y_train)

    print("\nTop 10 configuraciones:")
    resultados = pd.DataFrame(grid.cv_results_)
    display(
        resultados
        .filter(regex="(param.*|mean_t|std_t)")
        .drop(columns="params")
        .sort_values("mean_test_score", ascending=False)
        .head(10)
    )

    return grid


# =============================================================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# =============================================================================

def entrenar_modelo_final(X_train_scaled, y_train) -> MLPClassifier:
    """
    Entrena el modelo final con la mejor configuración hallada.

    Configuración seleccionada tras RandomizedSearchCV:
    - hidden_layer_sizes (5, 10): dos capas ocultas que captan relaciones
      no lineales sin sobrecomplicar la arquitectura.
    - learning_rate_init 0.01: convergencia estable sin oscilaciones.
    - solver 'adam': eficiente para datasets de tamaño mediano.
    - alpha 1: regularización L2 moderada para controlar el sobreajuste.
    - max_iter 1000: iteraciones suficientes para la convergencia.
    """
    modelo = MLPClassifier(
        hidden_layer_sizes=(5, 10),
        learning_rate_init=0.01,
        solver="adam",
        alpha=1,
        max_iter=1000,
        random_state=123
    )

    modelo.fit(X=X_train_scaled, y=y_train)
    return modelo


# =============================================================================
# 5. EVALUACIÓN
# =============================================================================

def evaluar_modelo(modelo, X_train_scaled, X_test_scaled, y_train, y_test):
    """Imprime métricas y muestra la matriz de confusión."""
    predicciones  = modelo.predict(X_test_scaled)
    train_score   = modelo.score(X_train_scaled, y_train)
    test_score    = modelo.score(X_test_scaled, y_test)
    accuracy      = accuracy_score(y_test, predicciones)
    null_acc      = y_test.value_counts(normalize=True).max()

    print(f"Train Score  : {train_score:.4f}")
    print(f"Test Score   : {test_score:.4f}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Null Accuracy: {null_acc:.4f}")
    print("El modelo ES mejor que el azar."   if test_score > null_acc          else "El modelo NO supera el azar.")
    print("Sobreajuste detectado."             if train_score > test_score + 0.05 else "Sin sobreajuste significativo.")

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, predicciones))

    etiquetas = sorted(y_test.unique())
    conf_matrix = confusion_matrix(y_test, predicciones)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="PuBu",
        xticklabels=etiquetas, yticklabels=etiquetas
    )
    plt.title("Matriz de Confusión — Modelo Final")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    df = cargar_datos("data/clean_drugs_classif.csv")

    X_train_scaled, X_test_scaled, y_train, y_test = preparar_datos(df)

    buscar_hiperparametros(X_train_scaled, y_train)

    modelo_final = entrenar_modelo_final(X_train_scaled, y_train)

    evaluar_modelo(modelo_final, X_train_scaled, X_test_scaled, y_train, y_test)
