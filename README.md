# Detector Númerico

En este repositorio es posible encontrar un modelo sencillo y muy fácil de comprender capaz de realizar detecciones númericas de un solo digito. El modelo consiste en una red convolucional sencilla implementada con la libreria keras y cuenta además con los pesos de entrenamiento y la arquitectura en formato JSON.
Detalles

Este modelo fue mi primer acercamiento con las redes convolucionales además de mis primeros pasos utilizando Keras. Fue entrenado utilizando el dataset MNist, el cual, a pesar de brindar excelentes metricas tras su entrenamiento, en la práctica, me fue posible detectar que le era dificultoso el predecir eficientemente con imagenes de entorno real. Tras algunas pruebas, detecté que este dataset tenía todas sus imagenes con un fondo negro, lo cual podría ser una causa a la hora de predecir en un entorno real, por lo que tras duplicar el mismo dataset y haber invertido su tonalidad de grises, la presición de predicción con imagenes reales aumento considerablemente.
