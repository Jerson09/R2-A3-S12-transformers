# R2-A3-S12 — Mecanismos de Atención y Transformers aplicados a Series de Tiempo

Actividad individual de la asignatura **604013 Deep Learning – Conceptos**  
Especialización en Inteligencia Artificial · Universidad de Cundinamarca  
Autor: Jerson Baquero Cháves

## Objetivo

Implementar desde cero un mecanismo de atención por producto punto escalado y una arquitectura Transformer (encoder), aplicados al pronóstico un paso adelante de una serie de tiempo univariada, y evaluar su desempeño frente a una red recurrente LSTM y a un predictor estadístico ingenuo.

## Contenido

- `notebooks/R2_A3_S12_Transformers.ipynb` — implementación completa, entrenamiento, evaluación y análisis de interpretabilidad.

## Datos

Serie sintética de demanda eléctrica diaria (1.095 observaciones, tres años) construida con tendencia lineal, estacionalidad anual, estacionalidad semanal, efecto de fin de semana y ruido gaussiano de desviación 1,2. Al conocerse el proceso generador, el error irreducible queda determinado: MAE ≈ 0,958. Esto permite verificar objetivamente cuánta estructura real captura cada modelo.

## Metodología

- Ventana de entrada L = 30 días, horizonte H = 1.
- Partición cronológica 70/15/15 (745 / 160 / 160 ventanas), sin mezcla aleatoria para evitar fuga de información.
- Estandarización ajustada únicamente con el tramo de entrenamiento.
- Transformer: d_model 64, 4 cabezas, 2 capas de encoder, feed-forward 128, dropout 0,1.
- Entrenamiento con Adam (lr 1e-3), pérdida MSE, recorte de gradiente y parada temprana con paciencia de 20 épocas.
- El LSTM de comparación se entrena con la misma función, presupuesto y criterio de parada.

## Resultados

Conjunto de prueba (160 días), en unidades originales de la serie:

| Modelo | MAE | RMSE | MAPE (%) | Parámetros | Épocas |
| --- | --- | --- | --- | --- | --- |
| Transformer | 1,1158 | 1,4233 | 1,6753 | 67.265 | 73 |
| **LSTM** | **1,0214** | **1,2962** | **1,5247** | 17.217 | 134 |
| Ingenuo (rezago 7) | 1,4731 | 1,8505 | 2,1985 | 0 | — |
| *Piso teórico (ruido)* | *0,9575* | *1,2000* | — | — | — |

Ambos modelos aprendidos superan al baseline ingenuo: el LSTM reduce el MAE un 30,7 % y el Transformer un 24,3 %. El LSTM obtiene el menor error con menos de la cuarta parte de los parámetros del Transformer, resultado coherente con el régimen del experimento: serie univariada, ventana corta y conjunto de entrenamiento pequeño, donde la recurrencia constituye un sesgo inductivo más eficiente.

## Interpretabilidad

El perfil de atención de la última posición de la ventana concentra el 46,6 % del peso en los rezagos de 5 a 8 días, con máximo en los rezagos 6 y 7, y una segunda banda en los rezagos 12 a 14. Si la atención fuera uniforme, cada una de las 30 posiciones recibiría 0,0333. El modelo infirió la periodicidad semanal de la serie sin que se le indicara en ninguna parte.

## Reproducción

Abrir el notebook en Google Colab y ejecutar todas las celdas. No requiere GPU: el entrenamiento completo toma alrededor de 100 segundos en CPU. La semilla está fijada en 42.

## Referencias

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.
