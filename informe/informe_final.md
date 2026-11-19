# Clasificación automática de noticias: enfoque clásico, arquitectura propia y Transformer, y su generalización a fuentes no vistas

**Curso:** Deep Learning · **Informe final** · Proyecto final
**Integrantes:** Emmanuel Alvarez · Samuel Herrera · Cristian Cabarcas

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
| TF-IDF + Softmax | Regresión softmax (lineal) en PyTorch | Ninguno | ≈ 0,2 M (50.000 × 4 + 4) |
| BiLSTM + atención | Recurrente, entrenada desde cero | Ninguno | ≈ 9,3 M |
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
| Épocas | hasta 12–15 con early stopping (paciencia 3); en la corrida final se usaron 8 (BiLSTM) y 7 (NewsHybridNet) | 3 |
| Regularización | dropout, recorte de gradiente 1,0 | dropout, recorte de gradiente 1,0 |

Selección del mejor modelo por **macro F1 de validación**. Se guarda checkpoint tras cada época (con
reanudación automática).

**Curvas de aprendizaje.** Ver `models/results/curvas_aprendizaje.png`. Mejor época por macro F1 de
validación (corrida local final): TF-IDF + Softmax, época 3 de 6 (F1 val 0,919); BiLSTM, época 5 de 8
(0,920); NewsHybridNet, época 4 de 7 (0,929); DistilBERT, época 3 de 3 (0,949). Las líneas base y
NewsHybridNet alcanzan su mejor punto a mitad del entrenamiento y después se estancan, y DistilBERT
sigue mejorando hasta la última época. La pérdida de entrenamiento de BiLSTM y NewsHybridNet (≈ 0,38)
es mayor que la de validación (≈ 0,28–0,30) porque se entrena con *label smoothing* 0,1 y dropout, y
la de validación no los usa; no indica subajuste.

**Recursos computacionales.** Entrenamiento local en Windows 11, GPU NVIDIA GeForce RTX 5070 Ti
(17 GB), CPU AMD de 16 núcleos, 33 GB de RAM, Python 3.11 y PyTorch 2.11 (CUDA 12.8). Tiempo total
de entrenamiento: TF-IDF + Softmax 0,8 min (6 épocas), BiLSTM 2,5 min (8), NewsHybridNet 4,3 min (7) y
DistilBERT 6,5 min (3). Para comparación, la corrida anterior en Kaggle (Tesla T4) tardó 2,5, 6,0, 4,0 y
20,5 min respectivamente. El preprocesamiento corre en un solo hilo (`num_workers=0`), por eso las
redes pequeñas casi no se aceleran respecto a la T4.

---

## 4. Resultados

### 4.1 Desempeño en el test de AG News (en distribución)

| Modelo | Accuracy | Macro F1 |
| --- | --- | --- |
| TF-IDF + Softmax | 0,9191 | 0,9190 |
| BiLSTM + atención | 0,9212 | 0,9211 |
| NewsHybridNet | 0,9286 | 0,9283 |
| DistilBERT | 0,9464 | 0,9465 |

**Análisis por clase.** Las categorías más fáciles son Sports (F1 entre 0,969 y 0,988) y World (0,920
a 0,956). Las más difíciles son Business (0,886 a 0,917) y Sci/Tech (0,899 a 0,925) en los cuatro
modelos, y se confunden entre sí: noticias de empresas tecnológicas (por ejemplo, sobre contrataciones
de IBM o resultados de Dell) aparecen etiquetadas como Sci/Tech en el corpus, pero los modelos las
asignan a Business. También hay errores de World hacia Business (política y economía de un país) y
de World hacia Sports (cobertura de los Juegos Olímpicos). Estos errores se repiten en los cuatro
modelos, lo que sugiere que parte de ellos se debe a la ambigüedad de las etiquetas y no a la
capacidad del modelo. Las matrices de confusión normalizadas por modelo se generan en
`models/evaluate_models.ipynb`.

### 4.2 Generalización a datos externos (BBC News)

| Modelo | Macro F1 AG News | Macro F1 BBC (con politics) | Brecha con politics | Macro F1 BBC (sin politics) | Brecha sin politics |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + Softmax | 0,9190 | 0,9075 | 0,012 | 0,9755 | −0,057 |
| BiLSTM + atención | 0,9211 | 0,8619 | 0,059 | 0,9604 | −0,039 |
| NewsHybridNet | 0,9283 | 0,9001 | 0,028 | 0,9456 | −0,017 |
| DistilBERT | 0,9465 | 0,8511 | 0,095 | 0,9567 | −0,010 |

Brecha = macro F1 en AG News menos macro F1 en BBC. "Con politics" usa 1.015 ejemplos y 4 clases;
"sin politics" usa 773 ejemplos y 3 clases (el macro F1 se calcula solo sobre las clases presentes).

Análisis del desempeño:

- **Con `politics` (escenario más exigente) el rendimiento se degrada, y más cuanto más potente es el
  modelo en AG News.** DistilBERT, el mejor en distribución, pierde 0,095 puntos y queda por debajo de
  TF-IDF (0,851 frente a 0,908). El orden de AG News no se mantiene fuera de distribución.
- **NewsHybridNet es el segundo más robusto** (brecha 0,028, tras TF-IDF con 0,012) y mejor que BiLSTM
  y DistilBERT. Frente a BiLSTM, que parte de un F1 similar, la combinación de embeddings congelados y
  rama de caracteres podría ayudar, pero la ablación de la sección 4.4 no confirma el aporte de la
  rama B de forma aislada.
- **La clase `politics` explica la mayor parte de la caída.** BBC mezcla política interna del Reino
  Unido, que se mapea a World, con las noticias internacionales de AG News. Sin esa clase, los cuatro
  modelos obtienen en BBC un F1 mayor que en AG News (brechas negativas), porque quedan tres clases
  temáticamente más separables (deportes, negocios y tecnología). Las brechas sin `politics` no son
  comparables con las de AG News, que tiene cuatro clases.
- **Cautela.** No debe leerse que TF-IDF "generaliza mejor" en general: con 1.015 ejemplos y categorías
  definidas por otro medio, la diferencia entre modelos fuera de distribución es sensible a cómo se
  mapean las clases.

### 4.3 Costo computacional (latencia de inferencia)

| Modelo | ms/noticia (batch 1) | ms/noticia (batch 32) |
| --- | --- | --- |
| TF-IDF + Softmax | 0,31 | 0,15 |
| BiLSTM + atención | 1,33 | 0,10 |
| NewsHybridNet | 2,01 | 0,29 |
| DistilBERT | 2,11 | 0,70 |

Medido en la GPU RTX 5070 Ti (no se midió en CPU). Con lote 1 NewsHybridNet y DistilBERT tienen
latencia similar; con lote 32 NewsHybridNet es unas 2,4 veces más rápido por noticia.

Protocolo: 50 ejecuciones de calentamiento descartadas, mediana de 200 repeticiones, incluyendo
tokenización.

### 4.4 Ablaciones de NewsHybridNet

| Variante | Accuracy | Macro F1 |
| --- | --- | --- |
| Embeddings aleatorios (entrenables) | 0,9251 | 0,9250 |
| Sin rama léxica B | 0,9367 | 0,9367 |
| Completa (fastText congelado + rama B) | 0,9286 | 0,9283 |

El preentrenamiento léxico aporta poco en AG News: los embeddings aleatorios entrenables quedan solo
0,003 por debajo de la versión completa. **Resultado contrario a la hipótesis de diseño:** quitar la
rama léxica B *mejora* el modelo en 0,008 de macro F1, es decir, la rama de n-gramas de caracteres no
ayuda en AG News (con una sola semilla por variante). La compuerta tampoco se desplaza fuera de
distribución: el valor medio de `g` (peso de la rama semántica) es 0,355 en AG News y 0,354 en BBC. La
ablación no se evaluó en el conjunto externo, así que no se puede afirmar que la rama B no aporte
robustez ante palabras nuevas.

### 4.5 Robustez estadística

- **Multi-semilla (NewsHybridNet, semillas 30, 42 y 123):** accuracy 0,9286 / 0,9309 / 0,9283
  (media 0,9293 ± 0,0014) y macro F1 0,9283 / 0,9308 / 0,9280 (media 0,9291 ± 0,0016; desviación
  muestral). La variación entre semillas es de unas 3 décimas de punto.
- **Significancia (McNemar, test de AG News).** Todas las diferencias son significativas (p < 0,05)
  salvo TF-IDF frente a BiLSTM (p = 0,48): NewsHybridNet supera a TF-IDF (p = 6,5·10⁻⁴) y a BiLSTM
  (p = 0,014), y DistilBERT supera a NewsHybridNet (p = 5,6·10⁻¹²) y a todos los demás.

---

## 5. Conclusiones

- **Cumplimiento de objetivos.** Se construyeron y compararon los tres niveles de conocimiento previo
  (TF-IDF + Softmax y BiLSTM como líneas base, NewsHybridNet como arquitectura propia y DistilBERT
  como Transformer) con las mismas particiones de AG News. Se midió el desempeño en distribución
  (accuracy y macro F1), el costo (latencia y tiempo de entrenamiento), la generalización a un
  conjunto externo (BBC News) y el aporte de cada componente mediante ablaciones, con tres semillas
  y pruebas de McNemar. Además, los cuatro modelos están disponibles en una demo interactiva.
- **Hallazgo principal (responde la pregunta de investigación).** En distribución, el
  preentrenamiento profundo da una ventaja clara y estadísticamente significativa: DistilBERT alcanza
  0,9465 de macro F1 frente a 0,9283 de NewsHybridNet (1,8 puntos). Fuera de distribución la imagen
  cambia: con la clase `politics`, DistilBERT es el modelo que más se degrada (brecha de 0,095, hasta
  0,851 en BBC) y queda por debajo de TF-IDF (0,908) y de NewsHybridNet (0,900). NewsHybridNet pierde
  0,028 y conserva el segundo mejor resultado externo con una fracción del costo de DistilBERT. La
  mayor parte de la caída se debe a la definición de la clase `politics` (política interna del Reino
  Unido frente a World internacional): sin ella, los cuatro modelos obtienen en BBC un F1 entre 0,946
  y 0,976. Por tanto, el preentrenamiento profundo explica la ventaja dentro de AG News, pero no
  garantiza mejor generalización a otras fuentes.
- **La rama léxica no cumplió su propósito.** La hipótesis de diseño era que los n-gramas de
  caracteres darían robustez ante palabras nuevas, y que la compuerta pasaría peso a esa rama fuera de
  distribución. Ninguna de las dos se confirmó: quitar la rama B mejora el modelo en AG News (macro F1
  0,9367 frente a 0,9283), y el valor medio de la compuerta es el mismo en AG News (0,355) y en BBC
  (0,354). Los embeddings fastText congelados aportan poco frente a embeddings aleatorios entrenables
  (0,9283 frente a 0,9250). Con la evidencia disponible, la arquitectura más simple, sin rama léxica,
  es mejor opción que NewsHybridNet completa.
- **Compromiso desempeño/costo.** DistilBERT gana 1,8 puntos de macro F1 sobre NewsHybridNet, pero a
  mayor costo: con lote 32 su latencia por noticia es unas 2,4 veces mayor (0,70 frente a 0,29 ms) y
  tiene unas 7 veces más parámetros (≈ 67 M frente a ≈ 9,6 M entrenables). Con lote 1 la latencia de
  ambos es parecida (≈ 2 ms), y en la GPU utilizada el tiempo de entrenamiento también lo es (6,5
  frente a 4,3 min). TF-IDF + Softmax es el más rápido (0,31 ms con lote 1) y solo queda 2,7 puntos por
  debajo de DistilBERT, lo que lo hace competitivo cuando importa el costo. NewsHybridNet solo
  compensa frente a DistilBERT si se necesita servir muchas noticias por lote o con menos memoria.
- **Limitaciones.** Conjunto externo pequeño (1.015 ejemplos con `politics`, 773 sin ella), con otro
  balance de clases y otra definición de categorías (`politics`), por lo que las diferencias entre
  modelos fuera de distribución son sensibles al mapeo. La latencia se midió solo en GPU, no en CPU.
  Las ablaciones usan una sola semilla por variante y no se evaluaron en el conjunto externo; la
  multisemilla (3 semillas) solo cubre NewsHybridNet. El `hash()` de Python no es determinista entre
  procesos, por lo que los n-gramas se calculan con `zlib.crc32` y los pesos entrenados con la versión
  anterior no son reutilizables.
- **Trabajo futuro.**
  - Evaluar la ablación sin rama B en el conjunto externo y con más semillas, para confirmar si de
    verdad no aporta robustez.
  - Probar un ensamble de NewsHybridNet y DistilBERT.
  - Medir la latencia en CPU para sustentar el uso sin GPU.
  - Construir un conjunto externo con titulares recientes etiquetados a mano, con las mismas
    categorías que AG News, para evitar el problema de `politics`.

---

## Referencias

- Bahdanau, D., Cho, K. y Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Bojanowski, P., Grave, E., Joulin, A. y Mikolov, T. (2017). *Enriching Word Vectors with Subword Information*. TACL.
- Joulin, A., Grave, E., Bojanowski, P. y Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. EACL.
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP.
- Sanh, V., Debut, L., Chaumond, J. y Wolf, T. (2019). *DistilBERT, a distilled version of BERT*. NeurIPS EMC² Workshop.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
