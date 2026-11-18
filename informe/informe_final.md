# Clasificación automática de noticias: enfoque clásico, arquitectura propia y Transformer, y su generalización a fuentes no vistas

**Curso:** Deep Learning · **Informe final** · Proyecto final
**Integrantes:** Emmanuel Alvarez · Samuel Herrera · Cristian Cabarcas

> **Nota de edición (borrador):** los campos entre `[[ ... ]]` se completan con los resultados del
> entrenamiento. Las figuras y tablas marcadas con `TODO` se insertan desde los archivos generados por
> el notebook (`curvas_aprendizaje.png`, `recursos_computacionales.json`, `*.csv`) y por
> `evaluate_models.ipynb` (matriz de confusión, latencia, BBC). El documento final debe quedar en
> **PDF, máximo 3 páginas**, así que al completar hay que recortar texto y elegir solo las figuras más
> informativas.

---

## 1. Objetivos del proyecto

**Problema.** Los medios publican miles de noticias al día y clasificarlas por tema (mundo, deportes,
negocios, ciencia/tecnología) de forma manual no escala. Un clasificador puede obtener métricas altas
en el conjunto de prueba de su propio dataset y aun así fallar con noticias de otros medios, porque
cambian el estilo, el vocabulario y la época. El proyecto no busca solo buen desempeño: busca **medir
de forma honesta si los modelos generalizan fuera de la distribución de entrenamiento**.

**Pregunta de investigación.** ¿Cuánto del desempeño fuera de distribución se explica por el
preentrenamiento contextual profundo (Transformer) y cuánto se recupera con una arquitectura ligera
que solo usa conocimiento léxico preentrenado (embeddings estáticos y n-gramas de caracteres)?

**Objetivo general.** Desarrollar y comparar un clasificador de noticias en cuatro categorías, con tres
niveles de conocimiento previo —línea base clásica (TF-IDF + regresión softmax), arquitectura neuronal
propia (NewsHybridNet) y un Transformer ajustado (DistilBERT)—, evaluando desempeño, costo
computacional y capacidad de generalizar a fuentes no vistas durante el entrenamiento.

**Objetivos específicos.**

1. Construir la línea base, diseñar y entrenar NewsHybridNet y ajustar DistilBERT sobre AG News con las
   mismas particiones, para una comparación justa.
2. Evaluar con macro F1, accuracy y tiempo de inferencia, y analizar errores por clase (matriz de
   confusión y ejemplos mal clasificados).
3. Medir la generalización con un conjunto externo de fuentes distintas, cuantificar el aporte del
   preentrenamiento mediante ablaciones, y entregar una demo interactiva.

---

## 2. Base de datos

**Dataset.** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (`fancyzhx/ag_news`), derivado
de un corpus de más de un millón de artículos de más de 2.000 fuentes. En inglés.

| Aspecto | Valor |
| --- | --- |
| Ejemplos de entrenamiento | 120.000 |
| Ejemplos de prueba | 7.600 (1.900 por clase) |
| Clases (balanceadas) | World, Sports, Business, Sci/Tech |
| Campos | Texto (título + descripción) y etiqueta |

**Particiones.** Se separa un **10 % estratificado** del entrenamiento como validación (semilla fija),
dejando ~108.000 train / ~12.000 val; el test oficial (7.600) solo se usa al final. El vocabulario se
construye **solo** con la partición de entrenamiento.

**Preprocesamiento** (justificado por cada modelo):

- **Línea base (TF-IDF):** minúsculas y vectorización TF-IDF con unigramas y bigramas
  (`min_df=5`, `max_features=50.000`, `sublinear_tf`). Representación dispersa adecuada para un modelo
  lineal.
- **BiLSTM y NewsHybridNet:** tokenización por palabras conservando signos relevantes, truncado/relleno
  a 80 palabras (cubre titular + descripción). NewsHybridNet añade n-gramas de caracteres (n = 3–5) con
  *hashing* a 2¹⁷ cubetas, para robustez ante palabras nuevas.
- **DistilBERT:** tokenizador WordPiece propio, truncado/padding a 128 tokens; sin limpieza agresiva,
  porque el Transformer aprovecha el texto original.
- **Data augmentation:** no se aplica; el dataset es grande y balanceado.

**Datos externos (generalización).** Conjunto de **BBC News** con categorías mapeadas a las de AG News
(`business→Business`, `sport→Sports`, `tech→Sci/Tech`, `politics→World` condicional;
`entertainment` se excluye). Se reportan resultados **con y sin** la clase `politics`, porque la
política de BBC es mayoritariamente interna del Reino Unido mientras que World en AG News es
internacional.

---

## 3. Modelo implementado

Se comparan cuatro modelos que representan distintos niveles de conocimiento previo. Los dos primeros
son líneas base; NewsHybridNet es la arquitectura propuesta por el grupo y DistilBERT es la referencia
del estado del arte.

| Modelo | Tipo | Conocimiento previo | Parámetros (aprox.) |
| --- | --- | --- | --- |
| TF-IDF + Softmax | Regresión softmax (lineal) en PyTorch | Ninguno | `[[dim TF-IDF]]` |
| BiLSTM + atención | Recurrente, entrenada desde cero | Ninguno | `[[params]]` |
| NewsHybridNet | Doble rama (semántica + léxica) con fusión por compuerta | Léxico (fastText congelado) | ≈ 9,6 M entrenables |
| DistilBERT | Transformer preentrenado, fine-tuning completo | Contextual profundo | ≈ 66 M |

**NewsHybridNet (arquitectura propia).** Combina una **rama semántica secuencial** (embeddings fastText
de 300d **congelados** → adaptador → tres Conv1D multiescala con kernels 2/3/5 → BiGRU → atención
aditiva ‖ max-pooling) con una **rama léxica de subpalabras** (n-gramas de caracteres con *hashing* →
EmbeddingBag), fusionadas por una **compuerta aprendida** `g = σ(W·[h_A,h_B]+b)`. El diseño responde a
que los titulares son cortos y densos en señales locales (de ahí las convoluciones), el orden importa
(BiGRU + atención) y fuera de distribución aparecen palabras nunca vistas (embeddings congelados + rama
de caracteres). Referencias de base: Kim (2014), Bojanowski et al. (2017), Joulin et al. (2017),
Bahdanau et al. (2015).

**DistilBERT** (`distilbert-base-uncased`, Sanh et al., 2019): 6 bloques Transformer, 768 dimensiones,
12 cabezas; se añade una cabeza de clasificación de 4 salidas sobre el token `[CLS]` y se ajusta toda
la red. Modificación respecto al original: capa de clasificación nueva de 4 clases.

**Entrenamiento.**

| Hiperparámetro | NewsHybridNet / BiLSTM | DistilBERT |
| --- | --- | --- |
| Pérdida | CrossEntropy con label smoothing 0,1 | CrossEntropy |
| Optimizador | AdamW, lr 1e-3, weight decay 1e-2 | AdamW, lr 2e-5, weight decay 1e-2 |
| Scheduler | One-cycle / coseno | Lineal con warmup (10 %) |
| Lote | 128 | 32 |
| Épocas | 12–15 con early stopping (paciencia 3) | 3 con early stopping |
| Regularización | dropout, recorte de gradiente 1,0 | dropout, recorte de gradiente 1,0 |

Selección del mejor modelo por **macro F1 de validación**. Se guarda checkpoint tras cada época (con
reanudación automática).

**Curvas de aprendizaje.** `TODO: insertar curvas_aprendizaje.png.` Interpretación esperada (ajustar a
lo observado): la línea base y la BiLSTM convergen en pocas épocas; DistilBERT converge muy rápido (2–3
épocas) y debe vigilarse el sobreajuste al estilo de AG News; NewsHybridNet converge de forma estable
gracias a los embeddings congelados como regularizador. `[[comentar brecha train/val y época del mejor
checkpoint por modelo]]`.

**Recursos computacionales.** `[[hardware: CPU / GPU (p. ej. Kaggle T4) / RAM, desde
recursos_computacionales.json]]`. Tiempo de entrenamiento aproximado por modelo:
`[[tiempos desde recursos_computacionales.json]]`.

---

## 4. Resultados

### 4.1 Desempeño en el test de AG News (en distribución)

| Modelo | Accuracy | Macro F1 |
| --- | --- | --- |
| TF-IDF + Softmax | `[[acc]]` | `[[f1]]` |
| BiLSTM + atención | `[[acc]]` | `[[f1]]` |
| NewsHybridNet | `[[acc]]` | `[[f1]]` |
| DistilBERT | `[[acc]]` | `[[f1]]` |

`TODO: insertar matriz(es) de confusión más informativas (p. ej. del mejor modelo y de NewsHybridNet).`
Errores esperados a confirmar: confusión entre Business y Sci/Tech, y entre World y Business.

### 4.2 Generalización a datos externos (BBC News)

| Modelo | Macro F1 AG News | Macro F1 BBC (con politics) | Macro F1 BBC (sin politics) | Brecha |
| --- | --- | --- | --- | --- |
| TF-IDF + Softmax | `[[]]` | `[[]]` | `[[]]` | `[[]]` |
| BiLSTM + atención | `[[]]` | `[[]]` | `[[]]` | `[[]]` |
| NewsHybridNet | `[[]]` | `[[]]` | `[[]]` | `[[]]` |
| DistilBERT | `[[]]` | `[[]]` | `[[]]` | `[[]]` |

Análisis del desempeño (completar con lo observado): `[[¿se mantiene o se degrada el rendimiento fuera
de distribución? ¿la brecha de NewsHybridNet es menor que la de las líneas base gracias a su rama
léxica? ¿cuánto separa a DistilBERT del resto?]]`.

### 4.3 Costo computacional (latencia de inferencia)

| Modelo | ms/noticia (batch 1) | ms/noticia (batch 32) |
| --- | --- | --- |
| TF-IDF + Softmax | `[[]]` | `[[]]` |
| BiLSTM + atención | `[[]]` | `[[]]` |
| NewsHybridNet | `[[]]` | `[[]]` |
| DistilBERT | `[[]]` | `[[]]` |

Protocolo: 50 ejecuciones de calentamiento descartadas, mediana de 200 repeticiones, incluyendo
tokenización.

### 4.4 Ablaciones de NewsHybridNet

| Variante | Accuracy | Macro F1 |
| --- | --- | --- |
| Embeddings aleatorios (entrenables) | `[[]]` | `[[]]` |
| Sin rama léxica B | `[[]]` | `[[]]` |
| Completa (fastText congelado + rama B) | `[[]]` | `[[]]` |

La diferencia entre "aleatorios" y "completa" cuantifica el aporte del preentrenamiento léxico; "sin
rama B" aísla el aporte de los n-gramas de caracteres. `[[interpretar, sobre todo en el conjunto
externo]]`.

### 4.5 Robustez estadística

- **Multi-semilla (NewsHybridNet, 3 semillas):** accuracy `[[media ± desv]]`, macro F1 `[[media ± desv]]`.
- **Significancia (McNemar):** `[[resumen de pares con p < 0,05; indicar si la ventaja de DistilBERT
  sobre NewsHybridNet y la de NewsHybridNet sobre la línea base son significativas]]`.

---

## 5. Conclusiones

> Redactar al final, atadas a los objetivos y respaldadas por los resultados; evitar afirmaciones
> genéricas. Esquema sugerido:

- **Cumplimiento de objetivos.** `[[¿se construyeron y compararon los tres niveles con particiones
  justas? ¿se midió la generalización con datos externos?]]`.
- **Hallazgo principal (responde la pregunta de investigación).** `[[cuánto del desempeño fuera de
  distribución aporta el preentrenamiento profundo frente al léxico; si NewsHybridNet recupera parte
  de esa brecha a una fracción del costo de DistilBERT]]`.
- **Compromiso desempeño/costo.** `[[relación entre macro F1 y latencia; cuándo conviene NewsHybridNet
  (CPU, baja latencia) frente a DistilBERT]]`.
- **Limitaciones.** Conjunto externo con distinto balance de clases y definición de categorías
  (politics); `hash()` no determinista entre procesos mitigado con `PYTHONHASHSEED`; alcance de las
  ablaciones y número de semillas acotado por el tiempo de cómputo.
- **Trabajo futuro.** `[[ensamble NewsHybridNet + DistilBERT; titulares recientes etiquetados a mano;
  más semillas]]`.

---

## Referencias

- Bahdanau, D., Cho, K. y Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Bojanowski, P., Grave, E., Joulin, A. y Mikolov, T. (2017). *Enriching Word Vectors with Subword Information*. TACL.
- Joulin, A., Grave, E., Bojanowski, P. y Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. EACL.
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
- Sanh, V., Debut, L., Chaumond, J. y Wolf, T. (2019). *DistilBERT, a distilled version of BERT*. NeurIPS EMC² Workshop.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
