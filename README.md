# Alzheimer Detection through MRI Image Classification Using CNNs

This project implements a convolutional neural network (CNN) model to classify MRI images into four stages of Alzheimer's disease. By detecting early structural brain changes, this model aims to enhance clinical decision-making, optimize healthcare resource management, and improve patient outcomes through timely interventions.

## Extended Description

Alzheimer's disease poses significant challenges to healthcare systems globally, with late diagnosis leading to increased costs and reduced treatment efficacy. This project addresses these challenges by developing a CNN-based model that classifies MRI scans into four categories: Alzheimer's Disease (AD), Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), and Late Mild Cognitive Impairment (LMCI). The solution combines advanced preprocessing, data augmentation, and optimized CNN architectures to achieve high diagnostic accuracy.

The dataset includes MRI scans from patients across different disease stages. The model identifies subtle anatomical patterns, such as hippocampal atrophy, enabling early detection and precise classification. Results demonstrate robust performance, with actionable insights for both clinical and business applications in healthcare.

### Data

The dataset comprises **33,984 preprocessed MRI images** categorized as follows:
- **AD (Alzheimer Disease)**: 8,960 images showing marked cerebral atrophy.
- **CN (Cognitively Normal)**: 6,464 control group images.
- **EMCI (Early Mild Cognitive Impairment)**: 9,600 images with subtle structural changes.
- **LMCI (Late Mild Cognitive Impairment)**: 8,960 images with pronounced anatomical alterations.

**Preprocessing Steps**:
- **Resizing**: Standardized to 128×128 pixels.
- **Normalization**: Pixel values scaled to [0, 1].
- **Data Augmentation**: Applied rotations, zoom, brightness adjustments, and shear transformations using `ImageDataGenerator`.
- **Validation Split**: 80% training, 20% validation.

### Model Overview

A **CNN architecture** was designed to extract hierarchical features from MRI scans:
- **Convolutional Blocks**: Three blocks with 64→128 filters, ReLU activation, and MaxPooling for dimensionality reduction.
- **Regularization**: Dropout (20-50%) and Batch Normalization to prevent overfitting.
- **Global Average Pooling**: Reduces spatial dimensions before dense layers.
- **Output Layer**: Softmax activation for multi-class classification.

**Training Configuration**:
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss Function**: Categorical cross-entropy.
- **Callbacks**: Early Stopping, ReduceLROnPlateau, and Model Checkpoint.

### Model Evaluation and Performance

- **Training Accuracy**: 95.4% | **Validation Accuracy**: 93.9%.
- **Confusion Matrix Analysis**:
  - **CN**: Perfect classification (100% recall).
  - **AD/LMCI/EMCI**: Minor misclassifications, primarily between EMCI and LMCI due to anatomical similarities.
- **Class-Specific Metrics**:

| Category | Precision | Recall |
|----------|-----------|--------|
| AD       | 96%       | 98%    |
| CN       | 99%       | 100%   |
| EMCI     | 92%       | 88%    |
| LMCI     | 89%       | 89%    |

**Key Insight**: The model achieves high generalizability, with a macro-averaged F1-score of 0.94.

### Future Work

- **EMCI vs. LMCI Differentiation**: Improve feature extraction to reduce misclassification between these stages.
- **Real-Time Integration**: Deploy the model in hospital systems for automated diagnostic support.
- **Explainability**: Use Grad-CAM or SHAP to visualize region-specific model decisions.
- **Hyperparameter Tuning**: Optimize dropout rates and learning schedules for higher precision.

## Libraries Used

- **Keras/TensorFlow**: For CNN architecture and training.
- **Pandas/NumPy**: For data handling.
- **Matplotlib/Seaborn**: For visualizations (e.g., average class images, metric plots).
- **Scikit-learn**: For evaluation metrics (confusion matrix, F1-score).

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

**Attribution Required**: If you use this code, please credit this repository and its author.

## Contact

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

--------------------------------------------------------------------------------------------------------------------------------------

# Detección de Alzheimer mediante Clasificación de Imágenes de IRM usando Redes Neuronales Convolucionales (CNN)

Este proyecto implementa un modelo de red neuronal convolucional (CNN) para clasificar imágenes de resonancia magnética (IRM) en cuatro etapas de la enfermedad de Alzheimer. Al detectar cambios estructurales tempranos en el cerebro, este modelo busca mejorar la toma de decisiones clínicas, optimizar la gestión de recursos en salud y potenciar los resultados en pacientes mediante intervenciones oportunas.

## Descripción Extendida

El Alzheimer representa un desafío crítico para los sistemas de salud globales, donde un diagnóstico tardío incrementa costos y reduce la eficacia de los tratamientos. Este proyecto aborda estos retos mediante un modelo basado en CNN que clasifica imágenes de IRM en cuatro categorías: Enfermedad de Alzheimer (AD), Cognitivamente Normal (CN), Deterioro Cognitivo Leve Temprano (EMCI) y Deterioro Cognitivo Leve Tardío (LMCI). La solución combina preprocesamiento avanzado, aumento de datos y arquitecturas CNN optimizadas para lograr alta precisión diagnóstica.

El conjunto de datos incluye IRM de pacientes en distintas etapas de la enfermedad. El modelo identifica patrones anatómicos sutiles, como la atrofia del hipocampo, permitiendo detección temprana y clasificación precisa. Los resultados demuestran un rendimiento robusto, con aplicaciones prácticas tanto clínicas como empresariales en el sector salud.

### Datos

El conjunto de datos contiene **33,984 imágenes de IRM preprocesadas**, categorizadas como:  
- **AD (Enfermedad de Alzheimer)**: 8,960 imágenes con atrofia cerebral marcada.  
- **CN (Cognitivamente Normal)**: 6,464 imágenes del grupo control.  
- **EMCI (Deterioro Cognitivo Leve Temprano)**: 9,600 imágenes con cambios estructurales sutiles.  
- **LMCI (Deterioro Cognitivo Leve Tardío)**: 8,960 imágenes con alteraciones anatómicas pronunciadas.  

**Preprocesamiento**:  
- **Redimensionamiento**: Estandarizado a 128×128 píxeles.  
- **Normalización**: Valores de píxel escalados a [0, 1].  
- **Aumento de datos**: Rotaciones, zoom, ajustes de brillo y transformaciones de corte usando `ImageDataGenerator`.  
- **División de validación**: 80% entrenamiento, 20% validación.  

### Arquitectura del Modelo

Se diseñó una **arquitectura CNN** para extraer características jerárquicas de las IRM:  
- **Bloques Convolucionales**: Tres bloques con filtros 64→128, activación ReLU y MaxPooling para reducir dimensionalidad.  
- **Regularización**: Dropout (20-50%) y Batch Normalization para evitar sobreajuste.  
- **Global Average Pooling**: Reduce dimensiones espaciales antes de las capas densas.  
- **Capa de Salida**: Activación Softmax para clasificación multiclase.  

**Configuración del Entrenamiento**:  
- **Optimizador**: Adam (tasa de aprendizaje = 0.001).  
- **Función de Pérdida**: Entropía cruzada categórica.  
- **Callbacks**: Early Stopping, ReduceLROnPlateau y Model Checkpoint.  

### Evaluación y Rendimiento del Modelo

- **Precisión en Entrenamiento**: 95.4% | **Precisión en Validación**: 93.9%.  
- **Análisis de Matriz de Confusión**:  
  - **CN**: Clasificación perfecta (100% recall).  
  - **AD/LMCI/EMCI**: Pequeñas confusiones, principalmente entre EMCI y LMCI por similitudes anatómicas.  
- **Métricas por Clase**:  

| Categoría | Precisión | Recall |
|-----------|-----------|--------|
| AD        | 96%       | 98%    |
| CN        | 99%       | 100%   |
| EMCI      | 92%       | 88%    |
| LMCI      | 89%       | 89%    |

**Hallazgo Clave**: El modelo logra alta generalización, con un F1-score promedio macro de 0.94.

### Trabajo Futuro

- **Diferenciación EMCI vs. LMCI**: Mejorar extracción de características para reducir confusiones.  
- **Integración en Tiempo Real**: Implementar el modelo en sistemas hospitalarios para diagnóstico automatizado.  
- **Explicabilidad**: Usar Grad-CAM o SHAP para visualizar regiones clave en las decisiones del modelo.  
- **Optimización de Hiperparámetros**: Ajustar tasas de dropout y ciclos de aprendizaje para mayor precisión.  

## Bibliotecas Utilizadas

- **Keras/TensorFlow**: Para arquitectura CNN y entrenamiento.  
- **Pandas/NumPy**: Para manejo de datos.  
- **Matplotlib/Seaborn**: Para visualizaciones (ej: imágenes promedio, gráficos de métricas).  
- **Scikit-learn**: Para métricas de evaluación (matriz de confusión, F1-score).  

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.  

**Atribución Requerida**: Si usas este código, por favor menciona este repositorio y a su autor.  

## Contacto

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]  

--------------------------------------------------------------------------------------------------------------


# Predictive Model for International Flight Delays

This project implements a machine learning model designed to predict international flight delays, leveraging historical flight data and various key features. By predicting delays of 15 minutes or more, this model aims to provide actionable insights for airlines, airports, and passengers, improving operational efficiency and the passenger experience.

## Extended Description

Flight delays are a critical concern for the aviation industry, affecting airlines, passengers, and airport operations. The challenge lies in accurately forecasting delays so that airlines can take preemptive actions to mitigate their impact. This project addresses this issue by developing a machine learning model capable of predicting whether a flight will experience a delay of 15 minutes or more, based on a wide range of variables.

The dataset used in this project contains historical flight data, which includes flight schedules, airline information, weather conditions, and operational statistics. The model uses these features to train and predict potential delays, helping stakeholders make better decisions regarding flight operations, passenger communications, and resource allocation.

### Data

The dataset consists of historical flight records, including features like:
- **Airline Information**: The airline operating the flight.
- **Flight Schedule**: Including flight times, number of seats, and aircraft type.
- **Flight Status**: Whether the flight experienced a delay or was on time.
- **Weather Conditions**: Such as temperature, wind speed, and visibility at the time of departure and arrival.
- **Airports**: The locations of departure and destination airports.

### Model Overview

To solve this problem, I used a **supervised learning approach** where the task is to predict a binary classification outcome: whether a flight will be delayed by 15 minutes or more. Various machine learning models were tested, including:
- **Decision Tree Classifier**: A simple yet interpretable model that splits the data based on feature thresholds.
- **XGBoost**: A highly efficient and scalable gradient boosting model, which outperformed others in accuracy.
- **MLPClassifier**: A neural network-based model to capture complex patterns in the data.

### Model Evaluation and Performance

The performance of the models was evaluated using standard classification metrics:
- **ROC-AUC**: Measures the area under the ROC curve, providing an aggregate measure of performance across all classification thresholds.
- **Precision and Recall**: Evaluate how well the model identifies true positives and avoids false positives.
- **F1-score**: A balance between precision and recall.

Among all the models tested, **XGBoost** provided the best performance with a ROC-AUC score of **0.6961** on the training dataset and **0.6958** on the test dataset. This makes it the most suitable model for predicting flight delays in this context.

### Future Work

- **Model Optimization**: Further tuning of hyperparameters to improve the model's accuracy.
- **Real-time Prediction**: Deploying the model to predict delays in real-time, integrating it with live flight data for more accurate forecasting.
- **Feature Engineering**: Including additional features such as flight crew information, historical delay patterns, and external events like strikes or weather disruptions could improve prediction accuracy.
- **Explainability**: Using SHAP (Shapley Additive Explanations) values to interpret and explain the model's decisions for stakeholders.

## Libraries Used

- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning models and evaluation metrics.
- **XGBoost**: For building the gradient boosting model.
- **Matplotlib**: For visualizations.
- **Seaborn**: For statistical data visualization.
- **SHAP**: For model explainability and interpreting feature importance.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

**Attribution Required**: If you use this code, please credit this repository and its author.


## Contact

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

--------------------------------------------------------------------------------------------------------------------

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

**Atribución requerida**: Si usas este código, menciona este repositorio y su autor.

## Contacto


David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]


--------------------------------------------------------------------------------------------------------

# Analysis of Music Trends on Spotify through Clustering

This project applies clustering algorithms to analyze music trends on Spotify, using a dataset with various characteristics of songs. The objective is to uncover hidden patterns in the music data and group songs into meaningful clusters that can inform marketing strategies and improve the streaming experience.

## Extended Description

As the music streaming industry continues to grow, understanding user preferences and the characteristics of popular songs becomes increasingly valuable. By applying **clustering techniques** to Spotify data, this project identifies patterns in song features, such as genre, tempo, acoustic features, and popularity. These insights can help optimize music recommendations, create targeted marketing strategies, and enhance user engagement.

### Data

The dataset includes a variety of features about the songs on Spotify, including:
- **Song Features**: Such as tempo, duration, and key.
- **Popularity Metrics**: Metrics like the number of streams and likes.
- **Audio Features**: Including loudness, danceability, valence (happiness), and energy.
- **Genres**: Information about the genre of each song.

### Model Overview

In this project, two clustering algorithms were employed:
- **K-Means**: A widely used unsupervised learning algorithm that partitions data into a predefined number of clusters.
- **Gaussian Mixture Model (GMM)**: A probabilistic model that assumes all data points are generated from a mixture of several Gaussian distributions.

### Model Evaluation and Results

The clustering models were evaluated using:
- **Silhouette Score**: A metric to evaluate how similar each point is to its own cluster compared to other clusters.
- **Calinski-Harabasz Index**: Measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion.

Key findings from the analysis include:
- Distinct clusters representing different musical genres, such as classical acoustic songs and energetic pop hits.
- The K-Means algorithm performed well in identifying compact, well-separated clusters, while the GMM provided a more nuanced view with overlapping clusters that reflect real-world music trends.

### Future Work

- **Dynamic Clustering**: Implementing real-time clustering as new songs are added to the Spotify platform.
- **Personalization**: Developing personalized music recommendation systems based on user preferences and cluster profiles.
- **Market Segmentation**: Using clusters to identify specific listener segments and tailor marketing campaigns accordingly.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For clustering algorithms and evaluation metrics.
- **Matplotlib**: For visualizing clusters and song features.
- **Seaborn**: For advanced statistical visualizations.

## License

This project is distributed under the MID License. See the `LICENSE` file for more details.

**Attribution Required**: If you use this code, please credit this repository and its author.


## Contact

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]



-------------------------------------------------------------------------------------------------------------


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

**Atribución requerida**: Si usas este código, menciona este repositorio y su autor.

## Contacto

David Alejandro Garza Antuña - [davidonai312@gmail.com] - [+52 7229098161]

