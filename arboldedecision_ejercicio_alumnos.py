

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
df = pd.read_csv(url)


"""Lo primero que haremos es clasificar las casas por precio: muy alto, alto, medio, bajo y muy bajo. Para ello deberás convertir en categórica la columna "median_house_value". Esto lo haremos para que el problema se transforme en un problema de clasificación y no de regresión. Recuerda borrar la columna "median_house_value" con los valores numéricos."""


"""Divide el dataset para guardar un 10% a validación"""


"""# Ejercicio 1
 Implementa los transformadores que consideres para realizar el preprocesamiento de los datos.
"""

# Clasificar las casas por precio
df['median_house_value_category'] = pd.cut(df['median_house_value'],
                                            bins=[0, 75000, 150000, 225000, 300000, float('inf')],
                                            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

# Eliminar la columna original con valores numéricos
df.drop('median_house_value', axis=1, inplace=True)

# Mostrar el DataFrame resultante
df.head()

"""# Ejercicio 2

Crea el pipeline necesario para realizar el preprocesamiento de los datos y el entrenamiento del modelo utilizando arboles de decisión.
"""


# Dividir el conjunto de datos para la validación (10%)

train_set, val_set = train_test_split(df, test_size=0.1, random_state=42)

# Separar características y etiquetas
X_train = train_set.drop("median_house_value_category", axis=1)
y_train = train_set["median_house_value_category"]

# Definir columnas numéricas y categóricas
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(include=[np.object]).columns.tolist()

# Crear transformador numérico
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Crear transformador categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear preprocesador que aplica transformadores según el tipo de columna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Crear pipeline con preprocesador y clasificador RandomForest
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42))])

"""# Ejercicio 3

Prueba con diferentes hipeparámetros evaluando los modelos resultantes a través de la técnica de cross validation y seleccionando el que mejor exactitud tenga.
"""


# Probar diferentes hiperparámetros
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
}

# Usar GridSearchCV para encontrar los mejores hiperparámetros
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mostrar los resultados del mejor modelo
print("Mejores Hiperparámetros:", grid_search.best_params_)
print("Mejor Exactitud:", grid_search.best_score_)

#escribe aquí el que mejor exactitud tenga

"""# Ejercicio 4

Muestra la matriz de confusión que ofrece el pipeline que mejor exactitud tenga, para ello utiliza el subconjunto de datos para validación que has guardado al principio.
"""


# Obtener las predicciones en el conjunto de validación
X_val = val_set.drop("median_house_value_category", axis=1)
y_val = val_set["median_house_value_category"]

best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Crear y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Verdaderas")
plt.show()

"""# Ejercicio 5

Siguiendo los mismos pasos crea un modelo de regresión que permita predecir el precio medio de la vivienda. Para ello tendrás que utilizar el dataset con la columna "median_house_value" en forma numérica. Puedes descargarlo otra vez.
"""

