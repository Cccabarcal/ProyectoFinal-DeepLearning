# Guion de sustentación — Clasificador automático de noticias

**Formato:** 15 min de presentación + 5 min de preguntas. **Los 3 integran­tes exponen.**
**Reparto:** ~5 min por persona. Nombres a asignar: **A**, **B**, **C** (pónganlos ustedes).

> **Requisito explícito del profesor:** abrir el código y mostrar en vivo (1) la **arquitectura de los
> datos**, (2) el **ciclo de entrenamiento**, y (3) **dónde se congelan y descongelan** los embeddings.
> Este guion ya indica la **celda exacta** del notebook `models/train_models.ipynb` para cada punto.

> **Recordatorio del profesor:** cualquiera debe poder responder cualquier pregunta, sin importar qué
> sección presentó. Ensayar una ronda cruzada de preguntas entre los tres.

---

## Antes de empezar (montaje, 5 min antes)

- [ ] Abrir `models/train_models.ipynb` y dejar visibles las pestañas/celdas: **6, 16, 37, 39, 41, 47, 51**.
- [ ] Arrancar la interfaz y hacer **una predicción de calentamiento** (así DistilBERT ya está cargado
      y la demo responde al instante):
      ```powershell
      cd app\backend
      python -m uvicorn main:app --port 8000
      ```
      Abrir http://localhost:8000 y clasificar un titular cualquiera una vez.
- [ ] Tener a mano el PDF del informe y la presentación de diapositivas.
- [ ] Tener listos 3–4 titulares de ejemplo en inglés para la demo (uno por categoría).

---

## BLOQUE 1 — Persona A · Problema, objetivos y datos (≈5 min)

**Diapositivas**

1. **Problema y justificación.** Clasificar noticias por tema no escala manualmente; además un modelo
   puede ir muy bien en su propio test y fallar con otras fuentes (cambia estilo, vocabulario, época).
2. **Pregunta de investigación.** ¿Cuánto del desempeño fuera de distribución lo explica el
   preentrenamiento profundo (Transformer) y cuánto se recupera con conocimiento solo léxico
   (embeddings estáticos + n-gramas de caracteres)?
3. **Objetivo general + específicos** (leerlos tal cual del informe, sección 1).
4. **Base de datos: AG News.** 120.000 train / 7.600 test, 4 clases balanceadas (World, Sports,
   Business, Sci/Tech), en inglés. Partición: 10 % estratificado de train como validación → ~108k/12k;
   test solo al final; vocabulario construido **solo** con train.

**[ABRIR CÓDIGO] — Arquitectura de los datos** (requisito 1 del profe)

- **Celda 6** (sección "## 2. Carga del dataset AG News y partición train/val/test"): mostrar
  `train_test_split(..., test_size=0.10, stratify=train_label_all)`. Explicar: estratificado = mantiene
  el balance de clases; semilla fija = reproducible; test intacto.
- **Celda 37** (sección "### 7.2 … n-gramas de caracteres"): mostrar `char_ngram_hashes(...)` y la clase
  `AGNewsHybridDataset`. Explicar que cada noticia se convierte en **dos representaciones**: una secuencia
  de palabras (rama A) y una bolsa de n-gramas de caracteres con *hashing* (rama B). Esa es la
  "arquitectura de los datos" que entra a NewsHybridNet.

**Frase de cierre del bloque:** "Con estos datos entrenamos cuatro modelos con tres niveles de
conocimiento previo; los explica B."

---

## BLOQUE 2 — Persona B · Modelos, entrenamiento y congelar/descongelar (≈5 min)

**Diapositivas**

1. **Los 4 modelos y los 3 niveles de conocimiento previo:**
   - Nivel 0: TF-IDF + Softmax y BiLSTM desde cero (sin conocimiento previo).
   - Nivel 1: **NewsHybridNet** (léxico: fastText congelado + n-gramas de caracteres).
   - Nivel 2: DistilBERT (contextual profundo, fine-tuning completo).
2. **NewsHybridNet (nuestra arquitectura).** Doble rama: semántica (fastText congelado → adaptador →
   Conv1D 2/3/5 → BiGRU → atención ‖ max-pool) + léxica (n-gramas → EmbeddingBag), fusionadas por una
   **compuerta aprendida**. Referencias: Kim 2014, Bojanowski 2017, Joulin 2017, Bahdanau 2015.
3. **DistilBERT.** 6 bloques Transformer; cabeza nueva de 4 clases sobre `[CLS]`; se ajusta toda la red.
4. **Hiperparámetros** (tabla del informe): AdamW, label smoothing, lotes, épocas, early stopping.

**[ABRIR CÓDIGO] — Ciclo de entrenamiento** (requisito 2 del profe)

- **Celda 16** (sección "### 4.4 Bucle de entrenamiento"): mostrar `train_rnn` y señalar el bucle
  `for epoch in range(...)`: forward → `loss` → `loss.backward()` → `clip_grad_norm_` →
  `optimizer.step()` → evaluación en validación → `save_checkpoint` (checkpoint por época + early
  stopping por macro-F1). Explicar que **los 4 modelos siguen el mismo patrón**.
- (Opcional, si sobra tiempo) **Celda 41** (`train_hybrid`): mismo esquema para NewsHybridNet, con
  scheduler One-Cycle y `build_param_groups` (weight decay salvo en LayerNorm/bias).

**[ABRIR CÓDIGO] — Congelar / descongelar** (requisito 3 del profe, el que más le interesa)

- **Celda 39** (sección "### 7.3 Definición de NewsHybridNet"): señalar la línea
  `self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=True, padding_idx=pad_idx)`.
  Explicar: **`freeze=True` = embeddings fastText CONGELADOS**; no se actualizan en el entrenamiento, para
  que las palabras conserven su semántica general y no se sobreajusten a AG News (y los OOV se componen
  por subpalabras).
- **Celda 51** (sección "## 10. Ablaciones"): señalar `NewsHybridNetAblation(..., freeze_emb=False, ...)`,
  donde se **DESCONGELAN** (embedding entrenable, inicializado al azar). Explicar que esta ablación es la
  que mide cuánto aporta el preentrenamiento: comparar "congelado" vs "descongelado/aleatorio".

**Frase de cierre:** "Con esto entrenado, veamos qué resultados dio y si de verdad generaliza; sigue C."

---

## BLOQUE 3 — Persona C · Resultados, generalización y demo (≈5 min)

**Diapositivas**

1. **Resultados en AG News (en distribución):** tabla de accuracy y macro-F1.
   DistilBERT 0,946 > NewsHybridNet 0,928 > BiLSTM 0,921 > TF-IDF 0,919. Mostrar **curvas de aprendizaje**.
   - (Opcional **celda 47**, `plot_curves()`, si quieren mostrarlas desde el código.)
2. **Análisis de errores.** Las clases difíciles son Business y Sci/Tech y se confunden entre sí; parte
   del error es ambigüedad de etiquetas (empresas tecnológicas). Mostrar matriz de confusión.
3. **Generalización a BBC News (datos externos).** Tabla con/sin `politics`. Hallazgo clave y honesto:
   **con `politics`, DistilBERT es el que MÁS se degrada** (brecha 0,095) y queda por debajo de TF-IDF y
   NewsHybridNet. La clase `politics` (política interna de UK vs World internacional) explica casi toda la
   caída: **sin ella, los 4 modelos van incluso mejor en BBC.**
4. **Ablaciones (resultado contra la hipótesis).** Quitar la rama léxica B **mejora** el modelo en AG
   News; los embeddings congelados aportan poco frente a entrenables. Lo reportamos de forma honesta.
5. **Costo.** Latencia (DistilBERT ~2,4× más lento por noticia con lote 32) y recursos/tiempo de
   entrenamiento (RTX 5070 Ti; DistilBERT 6,5 min).

**[DEMO EN VIVO] — la interfaz**

- Pegar un titular (p. ej. uno de negocios y otro de tecnología) y mostrar las predicciones de los 4
  modelos en paralelo, con barras de probabilidad.
- Para **NewsHybridNet**, resaltar el **mapa de atención** (palabras que más pesaron) y el valor de la
  **compuerta** (semántica vs léxica).

**Conclusiones** (atarlas a los objetivos, de la sección 5 del informe):

- El preentrenamiento profundo gana en distribución, pero **no garantiza** mejor generalización externa.
- NewsHybridNet es competitiva y más barata, aunque su rama léxica no aportó lo esperado (lo decimos).
- Limitaciones: conjunto externo pequeño y sensible al mapeo de `politics`; latencia medida solo en GPU.

---

## Preguntas probables del profesor (prepárenlas los tres)

- **¿Por qué congelar los embeddings y no entrenarlos?** → Regularización: evita sobreajuste a AG News y
  conserva semántica general; además fastText compone OOV por subpalabras. La ablación lo compara.
- **¿Por qué DistilBERT generaliza peor en BBC?** → El fine-tuning completo lo ajusta al estilo de AG
  News; la clase `politics` de BBC (UK interno) no coincide con World (internacional). Sin esa clase, va bien.
- **¿Por qué la rama léxica no ayudó?** → En AG News los titulares comparten vocabulario con el train;
  la señal de n-gramas aporta poco. Habría que probar en el conjunto externo (trabajo futuro).
- **¿Qué es macro-F1 y por qué no solo accuracy?** → Promedia F1 por clase sin favorecer a ninguna;
  importa aunque las clases estén balanceadas, para ver el desempeño por categoría.
- **¿Cómo evitaron fuga de datos (data leakage)?** → Vocabulario y TF-IDF ajustados **solo** con train;
  test usado una única vez al final.
- **¿Qué significa la compuerta?** → Aprende, por dimensión y ejemplo, cuánto confiar en la rama
  semántica vs la léxica. Reportamos su valor medio (≈0,355).

---

## Checklist final de tiempos

| Bloque | Persona | Minutos | Debe mostrar en código |
| --- | --- | --- | --- |
| 1. Problema, objetivos, datos | A | ~5 | Celdas 6 y 37 (arquitectura de datos) |
| 2. Modelos, entrenamiento, freeze | B | ~5 | Celda 16 (ciclo), celda 39 (freeze), celda 51 (unfreeze) |
| 3. Resultados, generalización, demo | C | ~5 | Demo de la interfaz (+ opcional celda 47) |
| Preguntas | Los 3 | 5 | — |

> Las referencias de celda son del notebook `models/train_models.ipynb`. Si el número cambia tras editar
> el notebook, buscar por el texto: `train_test_split`, `char_ngram_hashes`, `for epoch in range`,
> `from_pretrained(emb_matrix, freeze=True`, `freeze_emb=False`.
