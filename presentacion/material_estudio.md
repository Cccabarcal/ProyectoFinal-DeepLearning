# Material de estudio — Clasificador automático de noticias (Deep Learning)

Documento de estudio para preparar la sustentación. Está escrito en lenguaje sencillo y narrativo, de
principio a fin, para que cualquiera del grupo pueda entender el proyecto completo y responder preguntas,
sin importar qué parte le toque exponer.

---

## 1. De qué trata el proyecto, en una frase

Construimos y comparamos cuatro modelos que clasifican noticias en inglés en cuatro temas —World
(mundo), Sports (deportes), Business (negocios) y Sci/Tech (ciencia y tecnología)— y medimos de forma
honesta no solo cuál acierta más, sino cuál sigue funcionando bien cuando le damos noticias de **otra
fuente distinta** a la que usamos para entrenar.

El punto central no es "qué modelo gana", sino responder una pregunta: **¿cuánto del buen desempeño con
noticias nuevas se debe a que un modelo fue preentrenado con enormes cantidades de texto (como los
Transformers), y cuánto se puede lograr con una red más ligera que solo usa conocimiento de palabras?**

---

## 2. El problema y por qué importa

Los medios publican miles de noticias al día. Clasificarlas a mano por tema no es práctico. Un
clasificador automático permite organizar, filtrar y recomendar en tiempo real.

Hay una trampa conocida: un modelo puede tener métricas altísimas en el conjunto de prueba de su propio
dataset y aun así equivocarse con noticias de otro periódico, porque cambian el estilo de redacción, el
vocabulario y hasta la época (el dataset que usamos es de mediados de los 2000, así que no conoce
empresas, productos ni eventos recientes). Por eso insistimos en medir la **generalización**: cómo se
comporta el modelo "fuera de su zona de confort".

---

## 3. Los datos (base de datos)

Usamos **AG News**, un dataset público muy conocido. Tiene 120.000 noticias para entrenar y 7.600 para
la prueba final, repartidas de forma equilibrada entre las cuatro clases. Cada ejemplo es el título más
una breve descripción, en inglés.

Dividimos los datos en tres partes, lo cual es una práctica estándar y algo que el profesor evalúa:

- **Entrenamiento (~108.000):** con lo que el modelo aprende.
- **Validación (~12.000):** separamos el 10 % del entrenamiento, de forma "estratificada" (es decir,
  manteniendo la proporción de cada clase). Sirve para ajustar y para decidir cuál es el mejor momento
  del entrenamiento, sin tocar la prueba.
- **Prueba (7.600):** el conjunto oficial, que **solo se usa una vez, al final**. Esto evita el
  "data leakage" (que el modelo haga trampa viendo datos de prueba durante el ajuste).

Un detalle importante: el vocabulario y las transformaciones se construyen **solo con el conjunto de
entrenamiento**, nunca con el de prueba. Esa es otra forma de evitar fuga de información.

Para medir la generalización usamos un conjunto externo, **BBC News**, de otra fuente. Mapeamos sus
categorías a las nuestras (business→Business, sport→Sports, tech→Sci/Tech, politics→World) y dejamos
fuera "entertainment" porque no tiene equivalente. La categoría "politics" la tratamos con cuidado,
porque la política de la BBC es sobre todo interna del Reino Unido, mientras que "World" en AG News es
internacional; por eso reportamos resultados con y sin ella.

---

## 4. Los cuatro modelos y la idea de "niveles de conocimiento previo"

La gracia del proyecto es comparar modelos que traen distinta cantidad de "conocimiento aprendido de
antemano". Es como una escalera:

- **Nivel 0 — sin conocimiento previo.** Todo lo aprenden desde cero con AG News.
  - **TF-IDF + Softmax:** el enfoque clásico. Representa cada texto por la frecuencia de sus palabras y
    un modelo lineal decide la categoría. Rápido e interpretable; es nuestra línea base.
  - **BiLSTM + atención:** una red recurrente que lee la secuencia de palabras en ambos sentidos y usa
    "atención" para fijarse en las palabras más decisivas. Se entrena completamente desde cero.

- **Nivel 1 — conocimiento léxico (de palabras).**
  - **NewsHybridNet:** nuestra arquitectura propia. Usa embeddings **fastText** (vectores de palabras ya
    entrenados con mucho texto) que mantenemos **congelados**, más una rama que mira fragmentos de
    caracteres. No conoce "contexto profundo", solo el significado general de las palabras.

- **Nivel 2 — conocimiento contextual profundo.**
  - **DistilBERT:** un Transformer preentrenado con grandes cantidades de texto; lo ajustamos
    (fine-tuning) a nuestras cuatro clases. Es el representante del estado del arte.

Al ordenar los modelos así, podemos atribuir las mejoras a cada tipo de conocimiento, en lugar de solo
comparar "clásico contra moderno".

---

## 5. NewsHybridNet explicada con palabras (nuestra arquitectura)

Es una red de **dos ramas** que se combinan al final:

- **Rama semántica (A):** convierte cada palabra en su vector fastText (congelado), y luego aplica
  convoluciones que detectan combinaciones de 2, 3 y 5 palabras (por ejemplo "interest rates" o
  "Champions League"), una red recurrente BiGRU que integra el orden, y una capa de atención que decide
  qué palabras pesan más.
- **Rama léxica (B):** parte el texto en fragmentos de caracteres (n-gramas, de 3 a 5 letras) y los
  resume. La idea era que, aunque aparezca una palabra nueva que el modelo nunca vio, reconozca partes
  familiares (como "-coin", "bio-", "-ware").
- **Compuerta de fusión:** un mecanismo que aprende, para cada ejemplo, cuánto confiar en la rama
  semántica frente a la léxica, y mezcla ambas antes de decidir la categoría.

Por qué la diseñamos así: los titulares son cortos y con señales locales fuertes (de ahí las
convoluciones), el orden importa ("Apple shares fall" es Business, "Apple unveils new chip" es Sci/Tech),
y fuera de distribución aparecen palabras nuevas (de ahí los embeddings congelados y la rama de
caracteres).

---

## 6. Qué significa "congelar" y "descongelar" (concepto clave que el profe quiere oír)

**Congelar** un embedding significa que esos números (los vectores que representan cada palabra) **no se
modifican durante el entrenamiento**. Se quedan tal como vinieron de fastText.

Lo hacemos a propósito por dos razones:
1. Actúa como un **regularizador**: evita que la red se sobreajuste al estilo particular de AG News y
   conserve el significado general de las palabras.
2. Para palabras que no están en el vocabulario, fastText las arma a partir de sus subpartes, así que
   incluso palabras raras reciben un vector razonable.

**Descongelar** sería lo contrario: dejar que esos vectores se entrenen y cambien. En nuestras
**ablaciones** probamos justamente una versión con embeddings aleatorios y entrenables (descongelados)
para medir cuánto aportaba realmente el preentrenamiento. En el código, la diferencia es una sola
palabra: `freeze=True` (congelado, la versión propuesta) frente a `freeze_emb=False` (descongelado, la
ablación).

---

## 7. Cómo se entrenaron (ciclo de entrenamiento y configuración)

Los cuatro modelos siguen el mismo patrón de entrenamiento, que es el ciclo clásico de deep learning:

1. Se toma un lote de noticias.
2. El modelo predice (forward).
3. Se mide el error con una función de pérdida (usamos entropía cruzada; en las redes propias, con
   "label smoothing", que suaviza las etiquetas y ayuda a no sobreconfiar).
4. Se calcula cómo ajustar los pesos (backpropagation) y se recorta el gradiente para estabilizar.
5. El optimizador (AdamW) actualiza los pesos.
6. Al final de cada época se evalúa en validación y se guarda un "checkpoint".

Detalles que conviene saber:
- Elegimos el **mejor modelo por el macro-F1 de validación**, no por el último.
- Usamos **early stopping**: si no mejora en varias épocas seguidas, paramos.
- Guardamos un checkpoint **tras cada época**, de forma que si la sesión se corta, el entrenamiento se
  reanuda solo. Esto fue útil porque el entrenamiento es largo.
- DistilBERT se entrena con precisión mixta para ir más rápido en GPU.

---

## 8. Qué es macro-F1 y por qué lo usamos

El **accuracy** es el porcentaje de aciertos totales. El **F1** combina precisión y exhaustividad de una
clase. El **macro-F1** promedia el F1 de las cuatro clases dándoles el mismo peso, sin importar su
tamaño. Lo usamos porque nos interesa que el modelo funcione bien en **todas** las categorías, no solo en
las fáciles; es la métrica principal del proyecto.

---

## 9. Resultados (las cifras que hay que saber)

### Desempeño en AG News (en distribución)

| Modelo | Accuracy | Macro-F1 |
| --- | --- | --- |
| TF-IDF + Softmax | 91,9 % | 0,919 |
| BiLSTM + atención | 92,1 % | 0,921 |
| NewsHybridNet | 92,9 % | 0,928 |
| DistilBERT | 94,6 % | 0,946 |

Lectura: DistilBERT es el mejor con sus propios datos, con una ventaja clara y estadísticamente
significativa (lo confirmamos con la prueba de McNemar). Las clases más difíciles son Business y
Sci/Tech, que se confunden entre sí (por ejemplo, noticias de empresas tecnológicas); eso les pasa a los
cuatro modelos, lo que sugiere que parte del error viene de la ambigüedad de las propias etiquetas.

### Generalización a BBC News (fuera de distribución) — el hallazgo más interesante

Cuando pasamos a noticias de la BBC, la historia cambia:

- **Con la clase "politics", el modelo que más se degrada es DistilBERT** (el mejor en AG News). Cae
  hasta 0,851 de macro-F1 y queda **por debajo** de TF-IDF (0,908) y de NewsHybridNet (0,900). Es decir,
  ser el mejor "en casa" no garantiza ser el mejor "fuera de casa".
- La mayor parte de esa caída la explica la clase "politics": la política de la BBC es interna del Reino
  Unido, mientras que "World" en AG News es internacional. **Sin esa clase, los cuatro modelos van
  incluso mejor en BBC que en AG News**, porque quedan tres temas más fáciles de separar.
- NewsHybridNet resultó ser de los más robustos fuera de distribución, con una fracción del costo de
  DistilBERT.

### Ablaciones (resultado honesto, contra nuestra hipótesis)

Probamos quitar componentes de NewsHybridNet para ver qué aportaba cada uno:

- Quitar la **rama léxica (B)** en realidad **mejoró** el modelo en AG News (0,937 frente a 0,928). O
  sea, la rama de n-gramas de caracteres no ayudó como esperábamos.
- Usar embeddings aleatorios entrenables en vez de fastText congelados bajó muy poco (0,925 frente a
  0,928): el preentrenamiento léxico aportó poco en este dataset.
- La compuerta tampoco se movió fuera de distribución (su valor medio fue casi idéntico en AG News y en
  BBC).

Reportamos esto tal cual, porque la honestidad científica es parte de lo que evalúa el profesor: la
hipótesis de diseño no se confirmó, y una versión más simple sin la rama léxica habría sido mejor.

### Costo y recursos

DistilBERT es el más preciso pero el más caro: unas 7 veces más parámetros y, con lotes grandes, unas
2,4 veces más lento por noticia que NewsHybridNet. TF-IDF es el más rápido y queda a solo 2,7 puntos de
DistilBERT. Entrenamos en una GPU NVIDIA RTX 5070 Ti; los tiempos fueron de menos de 1 minuto (TF-IDF) a
6,5 minutos (DistilBERT).

---

## 10. Conclusiones (atadas a los objetivos)

- Cumplimos el objetivo: construimos y comparamos los cuatro modelos con las mismas particiones y
  medimos desempeño, costo y generalización.
- **Respuesta a la pregunta de investigación:** el preentrenamiento profundo (DistilBERT) da ventaja
  clara dentro de AG News, pero **no garantiza** mejor generalización a otra fuente. NewsHybridNet,
  mucho más ligera, se defiende bien fuera de distribución.
- **Autocrítica:** la rama léxica de NewsHybridNet no cumplió su propósito; un diseño más simple habría
  rendido igual o mejor.
- **Limitaciones:** el conjunto externo es pequeño y sensible a cómo se mapea "politics"; la latencia se
  midió solo en GPU; las ablaciones usaron una sola semilla.

---

## 11. Preguntas de repaso (para practicar en grupo)

1. ¿Por qué separamos un conjunto de validación además del de prueba, y por qué el de prueba solo se usa
   al final?
2. ¿Qué significa que los embeddings estén "congelados" y qué ventaja aporta?
3. ¿Por qué DistilBERT, siendo el mejor en AG News, generaliza peor en BBC?
4. ¿Qué papel juega la clase "politics" en la caída de desempeño con BBC?
5. ¿Qué es el macro-F1 y por qué lo preferimos al accuracy a secas?
6. ¿Qué hace la "compuerta" de NewsHybridNet?
7. ¿Qué nos dijeron las ablaciones y por qué es importante reportarlas aunque contradigan nuestra
   hipótesis?
8. ¿Cómo evitamos la fuga de datos (data leakage) en el preprocesamiento?
9. ¿Cuál es el compromiso entre precisión y costo al elegir entre DistilBERT y NewsHybridNet?
10. ¿Qué pasos sigue el ciclo de entrenamiento de un modelo y qué es el early stopping?

---

## 12. Glosario rápido

- **AG News:** dataset de noticias en inglés, 4 clases, el que usamos para entrenar y probar.
- **Embedding:** representación numérica de una palabra como un vector.
- **fastText:** método que produce esos vectores y sabe componer palabras nuevas por sus subpartes.
- **Congelar/descongelar:** dejar fijos o permitir que se ajusten los pesos de una capa.
- **Fine-tuning:** ajustar un modelo preentrenado a una tarea específica.
- **Macro-F1:** promedio del F1 de todas las clases, con el mismo peso.
- **Generalización / fuera de distribución:** desempeño con datos de otra fuente no vista al entrenar.
- **Ablación:** quitar o cambiar un componente del modelo para medir cuánto aportaba.
- **Early stopping:** detener el entrenamiento cuando deja de mejorar en validación.
- **Data leakage:** error de dejar que el modelo "vea" información de prueba durante el ajuste.
