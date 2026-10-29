# Anteproyecto — Clasificador automático de noticias

**Curso:** Deep Learning · **Entrega preliminar** · Proyecto final
**Integrantes:** Emmanuel Alvarez, Samuel Herrera, Cristian Cabarcas

## 1. Título

**Clasificación automática de noticias con TF-IDF + Regresión Logística y un Transformer ajustado (fine-tuning): evaluación de la generalización a fuentes no vistas en entrenamiento.**

## 2. Planteamiento del problema

Los medios publican miles de noticias al día y organizarlas por tema (deportes, negocios, ciencia/tecnología, mundo) de forma manual es lento y poco escalable. La clasificación automática de texto permite etiquetar, filtrar y recomendar contenido en tiempo real.

Un modelo puede obtener métricas muy altas en el conjunto de prueba de su propio dataset y aun así fallar con noticias de otros medios, porque cambian el estilo, el vocabulario y la estructura de los titulares. Este proyecto no busca solo un buen accuracy: busca **comparar un enfoque clásico con uno basado en Deep Learning y medir de forma honesta si el modelo generaliza fuera de la distribución de entrenamiento**.

## 3. Objetivos

**Objetivo general**

Desarrollar y comparar un clasificador de noticias en cuatro categorías, basado en un modelo Transformer ajustado y en una línea base clásica (TF-IDF + regresión logística), evaluando su desempeño y su capacidad de generalizar a noticias de fuentes no vistas durante el entrenamiento.

**Objetivos específicos**

1. Construir una línea base (TF-IDF + regresión logística) y ajustar un Transformer preentrenado sobre AG News, usando los mismos conjuntos de entrenamiento, validación y prueba para que la comparación sea justa.
2. Evaluar ambos modelos con macro F1, accuracy y tiempo de inferencia, y analizar los errores por clase (matriz de confusión y ejemplos mal clasificados).
3. Medir la generalización con un conjunto externo de noticias de fuentes distintas a las de entrenamiento, y entregar una demo en la que el usuario ingresa un titular o texto y obtiene la categoría predicha.

## 4. Datos

| Aspecto | Descripción |
| --- | --- |
| Dataset | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (`fancyzhx/ag_news`), derivado del corpus AG de más de 1 millón de artículos de más de 2.000 fuentes. |
| Tamaño | 120.000 ejemplos de entrenamiento y 7.600 de prueba. |
| Clases (4, balanceadas) | World (mundo/política), Sports (deportes), Business (negocios), Sci/Tech (ciencia y tecnología). |
| Campos | Texto (título + descripción) y etiqueta. |

> **Nota:** AG News no tiene una clase "política" como tal. Las noticias políticas están dentro de **World**, por lo que se reporta así.

**Preprocesamiento y particiones**

- **Validación:** se separa un 10 % del entrenamiento (estratificado, semilla fija) para ajustar hiperparámetros. El conjunto de prueba oficial solo se usa al final.
- **Línea base:** minúsculas, limpieza de caracteres especiales y vectorización TF-IDF con unigramas y bigramas.
- **Transformer:** se usa el tokenizador propio del modelo, con truncado y padding a una longitud máxima (~128 tokens, suficiente para titular + descripción). No se aplica limpieza agresiva, porque el Transformer aprovecha el texto original.
- **Data augmentation:** no se prevé de entrada, porque el dataset es grande y balanceado. Se evaluará solo si el análisis de errores lo justifica.

**Datos externos (reto de generalización)**

AG News no incluye la fuente de cada noticia, así que no se puede dividir por fuente dentro del propio dataset. Para probar la generalización se usará un conjunto externo de otras fuentes, por ejemplo **BBC News**, con sus categorías mapeadas a las de AG News, además de titulares recientes recolectados por el grupo con etiquetado manual. Las categorías de BBC sin equivalente (p. ej. entertainment) se excluyen. La elección final del conjunto externo se confirmará al inicio de la experimentación.

## 5. Arquitectura y modelos

**Línea base — TF-IDF + regresión logística.** TF-IDF representa cada texto como un vector disperso según la frecuencia de sus términos, y la regresión logística multiclase lo clasifica. Es rápida, interpretable (se pueden ver las palabras más influyentes por clase) y sirve de referencia para ver cuánto aporta realmente el Deep Learning.

**Modelo principal — Transformer ajustado (fine-tuning).** Se parte de un modelo preentrenado de la librería Hugging Face (candidato principal: **DistilBERT**; alternativa: BERT-base) y se añade una cabeza de clasificación de 4 salidas sobre la representación del token `[CLS]`.

- **Por qué un Transformer:** la autoatención captura contexto y significado más allá de la presencia de palabras, y el preentrenamiento sobre grandes corpus ayuda a generalizar a vocabulario y estilos que no aparecen en AG News.
- **Por qué DistilBERT:** conserva gran parte del rendimiento de BERT con menos parámetros y menor tiempo de inferencia, lo que importa para la demo y para los recursos disponibles.
- **Referencias:** Vaswani et al. (2017), *Attention Is All You Need*; Devlin et al. (2019), *BERT*; Sanh et al. (2019), *DistilBERT*.
- **Modificaciones respecto al original:** capa de clasificación nueva de 4 clases, entrenada junto con el resto de la red (fine-tuning completo).

**Entrenamiento (previsto).** Entropía cruzada como función de pérdida, optimizador AdamW con learning rate pequeño (~2e-5), warmup, 2 a 4 épocas con early stopping según la pérdida de validación, y entrenamiento en GPU (Google Colab o equivalente). Se guardarán las curvas de loss y métricas de entrenamiento y validación para analizar sobreajuste y convergencia.

## 6. Evaluación

| Métrica | Para qué sirve |
| --- | --- |
| Macro F1 | Desempeño promedio sobre las cuatro clases, sin favorecer a ninguna. |
| Accuracy | Referencia general y comparación con la literatura. |
| Tiempo de inferencia | Costo por noticia (ms) en CPU y GPU, para medir el compromiso entre precisión y velocidad. |

Análisis adicionales:

- Matriz de confusión y errores típicos. Se espera confusión entre Business y Sci/Tech, y entre World y Business.
- Comparación de ambos modelos en el conjunto de prueba de AG News y en el conjunto externo.
- Caída de desempeño entre datos "en distribución" y "fuera de distribución" como medida de generalización.
- Si hay tiempo: separar titular solo vs. titular + cuerpo para ver cuánto aporta el texto largo.

## 7. Demo

Interfaz sencilla (Gradio o Streamlit) en la que el usuario pega un titular o el cuerpo de una noticia y recibe la categoría predicha con su probabilidad. Opcionalmente mostrará la predicción del modelo base junto a la del Transformer.

## 8. Plan de trabajo

| Etapa | Actividad |
| --- | --- |
| 1 | Exploración del dataset, particiones y línea base TF-IDF + LR. |
| 2 | Fine-tuning del Transformer y análisis de curvas de entrenamiento. |
| 3 | Evaluación en prueba y en el conjunto externo, análisis de errores. |
| 4 | Demo, informe de 3 páginas (PDF) y presentación de 15 minutos. |

**Fecha límite de informe y presentación:** miércoles 11 de noviembre de 2026.

## 9. Resultados esperados y riesgos

- **Esperado:** la línea base alcanza ~90–92 % de accuracy en AG News y el Transformer ~94–95 %. La diferencia debería crecer en el conjunto externo, que es donde se espera ver el aporte del preentrenamiento.
- **Riesgo 1:** el conjunto externo puede tener distinto balance de clases o definiciones de categoría distintas. *Mitigación:* reportar macro F1 y documentar el mapeo de categorías.
- **Riesgo 2:** recursos de cómputo limitados. *Mitigación:* usar DistilBERT, longitud máxima de 128 tokens y, si hace falta, un subconjunto del entrenamiento.
