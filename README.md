# Categorizador de Género Musical con Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-green?style=for-the-badge)

Este proyecto implementa una red neuronal convolucional (CNN) entrenada sobre el conjunto de datos **GTZAN** para clasificar fragmentos de audio de 30 segundos en 10 géneros musicales distintos.

---

## Descripción general

El pipeline procesa archivos de audio extraendo sus **Espectrogramas de Frecuencia de Mel** (_Mel Spectrograms_), los cuales funcionan como imágenes 2D que representan la distribución de frecuencia a lo largo del tiempo. Posteriormente, una arquitectura basada en **Redes Neuronales Convolucionales (CNN)** extrae los patrones visuales-acústicos y los conecta a capas densas para determinar el género final.

### Géneros Soportados (GTZAN Dataset)

`Blues` • `Classical` • `Country` • `Disco` • `Hiphop` • `Jazz` • `Metal` • `Pop` • `Reggae` • `Rock`

---

## Arquitectura del Proyecto

```text
.
├── data_understanding.py  # Análisis exploratorio y visualización de espectrogramas
├── data_preparation.py    # Extracción de características (Mel) y almacenamiento por lotes
├── data_modeling.py       # Pipeline de entrenamiento, evaluación y diagnóstico del modelo
├── models.py              # Definición de las arquitecturas de Deep Learning
├── test.py                # Inferencia sobre el set de prueba o audio externo
└── spotify_aux.py         # Módulo auxiliar para descarga e inferencia desde Spotify
```

Herramienta para extraer canciones de Spotify: [Sunnify](https://github.com/sunnypatell/sunnify-spotify-downloader/releases/tag/v2.5.1)

Uso académico
