
# Predictive Model for International Flight Delays

This project develops a machine learning model to predict international flight delays, using historical data and various relevant variables. The goal is to provide valuable information for airlines, airports, and passengers, improving operational decision-making and user experience.

## Overview

The model focuses on predicting whether a flight will experience a delay of 15 minutes or more (DEP_DEL15), based on features such as the number of seats, flight schedules, airline, and historical performance data. A supervised learning approach was used, training several models and selecting XGBoost due to its superior performance in key metrics such as ROC-AUC and accuracy.

## Methodology

1. **Data Preprocessing**: Handling missing values, removing low-variance variables, and transforming categorical variables using Weight of Evidence (WOE).
2. **Variable Importance Analysis**: Using Information Value (IV) to select the most relevant variables.
3. **Modeling and Hyperparameter Tuning**: Training models like Decision Tree Classifier, XGBClassifier, and MLPClassifier, with hyperparameter tuning through Grid Search and Randomized Search.
4. **Evaluation and Metrics**: Evaluating model performance using ROC-AUC, precision, recall, and F1-score.

## Key Results

- The XGBoost model showed the best performance with a ROC-AUC of 0.6961 on the training set and 0.6958 on the test set.
- A detailed analysis of the XGBoost model's hyperparameters was conducted, adjusting parameters like `colsample_bytree`, `max_depth`, and `learning_rate` to optimize performance.
- The model's stability was evaluated using the Population Stability Index (PSI), identifying variables that require continuous monitoring.
- SHAP values were calculated to interpret the influence of each variable on the model's predictions.

## Model Usage

The model is designed to be implemented in flight management systems, providing real-time predictions on possible delays. The user interface allows operators to make informed decisions to mitigate the impact of delays.

## Libraries Used

- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

## Contact

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

# Analysis of Music Trends on Spotify through Clustering

This project applies clustering algorithms to analyze music trends on Spotify, using a dataset with various song characteristics. The goal is to identify patterns and segment songs into meaningful groups to enhance personalization and marketing strategies in the streaming music industry.

## Overview

The project explores how clustering techniques can uncover hidden patterns in music and improve decision-making in the streaming industry. Algorithms like K-Means and Gaussian Mixture Model (GMM) are used to group songs based on their acoustic and popularity features, facilitating trend identification and user experience personalization.

## Methodology

1. **Data Preprocessing**: Data cleaning and transformation, including removing duplicates, handling outliers, and engineering new variables.
2. **Exploratory Data Analysis (EDA)**: Visualization and analysis of variable distributions and their correlations.
3. **Dimensionality Reduction**: Applying PCA to reduce data complexity and improve modeling efficiency.
4. **Model Selection and Evaluation**: Using K-Means and GMM to cluster songs, evaluating performance with metrics like silhouette score and Calinski-Harabasz index.
5. **Cluster Profiling**: Interpreting the resulting clusters to identify musical trends and audience profiles.

## Key Results

- Identification of distinct clusters representing different musical profiles, from classical acoustic songs to high-energy modern hits.
- Comparative evaluation of K-Means and GMM, highlighting their strengths and weaknesses in the context of music segmentation.
- Development of personalized marketing strategies based on identified clusters, improving the relevance of recommendations and user engagement.

## Model Usage

The model allows streaming platforms and music professionals to better understand user preferences and optimize content promotion. The generated clusters can be used to create thematic playlists, personalize recommendations, and design targeted marketing campaigns.

## Libraries Used

- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is distributed under the MID License. See the `LICENSE` file for more details.

## Contact

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]


-------------------------------------------------------------------------------------

# Modelo Predictivo de Retrasos en Vuelos Internacionales

Este proyecto desarrolla un modelo de aprendizaje automático para predecir retrasos en vuelos internacionales, utilizando datos históricos y diversas variables relevantes. El objetivo es proporcionar información valiosa para aerolíneas, aeropuertos y pasajeros, mejorando la toma de decisiones operativas y la experiencia del usuario.

## Descripción General

El modelo se enfoca en predecir si un vuelo sufrirá un retraso de 15 minutos o más (DEP_DEL15), basándose en características como el número de asientos, horarios de vuelo, aerolínea y datos históricos de rendimiento. Se utilizó un enfoque de aprendizaje supervisado, entrenando varios modelos y seleccionando XGBoost debido a su rendimiento superior en métricas clave como ROC-AUC y accuracy.

## Metodología

1.  **Preprocesamiento de Datos**: Tratamiento de valores perdidos, eliminación de variables de baja varianza y transformación de variables categóricas usando Weight of Evidence (WOE).
2.  **Análisis de Importancia de Variables**: Uso de Information Value (IV) para seleccionar las variables más relevantes.
3.  **Modelado y Selección de Hiperparámetros**: Entrenamiento de modelos como Decision Tree Classifier, XGBClassifier y MLPClassifier, con ajuste de hiperparámetros mediante Grid Search y Randomized Search.
4.  **Evaluación y Métricas**: Evaluación del rendimiento del modelo usando ROC-AUC, precisión, recall y F1-score.

## Resultados Clave

-   El modelo XGBoost mostró el mejor desempeño con un ROC-AUC de 0.6961 en el conjunto de entrenamiento y 0.6958 en el conjunto de prueba.
-   Se realizó un análisis detallado de los hiperparámetros del modelo XGBoost, ajustando parámetros como `colsample_bytree`, `max_depth` y `learning_rate` para optimizar el rendimiento.
-   Se evaluó la estabilidad del modelo mediante el Population Stability Index (PSI), identificando variables que requieren monitoreo continuo.
-   Se calcularon los SHAP values para interpretar la influencia de cada variable en las predicciones del modelo.

## Uso del Modelo

El modelo está diseñado para ser implementado en sistemas de gestión de vuelos, proporcionando predicciones en tiempo real sobre posibles retrasos. La interfaz de usuario permite a los operadores tomar decisiones informadas para mitigar el impacto de los retrasos.

## Librerías Utilizadas

-   Pandas
-   Scikit-learn
-   XGBoost
-   Matplotlib
-   Seaborn
-   SHAP

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto


David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

# Análisis de Tendencias Musicales en Spotify mediante Clusterización

Este proyecto aplica algoritmos de clusterización para analizar tendencias musicales en Spotify, utilizando un conjunto de datos con diversas características de canciones. El objetivo es identificar patrones y segmentar canciones en grupos significativos para mejorar la personalización y las estrategias de marketing en la industria del streaming musical.

## Descripción General

El proyecto explora cómo las técnicas de clusterización pueden revelar patrones ocultos en la música y mejorar la toma de decisiones en la industria del streaming. Se utilizan algoritmos como K-Means y Gaussian Mixture Model (GMM) para agrupar canciones basadas en sus características acústicas y de popularidad, facilitando la identificación de tendencias y la personalización de la experiencia del usuario.

## Metodología

1.  **Preprocesamiento de Datos**: Limpieza y transformación de datos, incluyendo la eliminación de duplicados, el tratamiento de outliers y la ingeniería de nuevas variables.
2.  **Análisis Exploratorio de Datos (EDA)**: Visualización y análisis de las distribuciones de variables y sus correlaciones.
3.  **Reducción de Dimensionalidad**: Aplicación de PCA para reducir la complejidad de los datos y mejorar la eficiencia del modelado.
4.  **Selección de Modelos y Evaluación**: Uso de K-Means y GMM para clusterizar canciones, evaluando el rendimiento con métricas como el coeficiente de silueta y el índice de Calinski-Harabasz.
5.  **Perfilamiento de Clústeres**: Interpretación de los clústeres resultantes para identificar tendencias musicales y perfiles de audiencia.

## Resultados Clave

-   Identificación de clústeres distintos que representan diferentes perfiles musicales, desde canciones acústicas clásicas hasta éxitos modernos de alta energía.
-   Evaluación comparativa de K-Means y GMM, destacando sus fortalezas y debilidades en el contexto de la segmentación musical.
-   Desarrollo de estrategias de marketing personalizadas basadas en los clústeres identificados, mejorando la relevancia de las recomendaciones y la participación del usuario.

## Uso del Modelo

El modelo permite a las plataformas de streaming y a los profesionales de la música comprender mejor las preferencias de los usuarios y optimizar la promoción de contenido. Los clústeres generados pueden utilizarse para crear playlists temáticas, personalizar recomendaciones y diseñar campañas de marketing dirigidas.

## Librerías Utilizadas

-   Pandas
-   Scikit-learn
-   Matplotlib
-   Seaborn

## Licencia

Este proyecto se distribuye bajo la licencia MID. Consulta el archivo `LICENSE` para más detalles.

## Contacto

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

