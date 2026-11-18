# Clasificador automático de noticias · Deep Learning

Proyecto final del curso de Deep Learning. Clasifica noticias de **AG News** en cuatro categorías
—**World, Sports, Business, Sci/Tech**— y compara tres niveles de conocimiento previo para medir
cuánto generaliza cada modelo a fuentes no vistas en el entrenamiento.

**Integrantes:** Emmanuel Alvarez · Samuel Herrera · Cristian Cabarcas

---

## ¿Qué contiene el proyecto?

El proyecto entrena **cuatro modelos** sobre el mismo dataset, con las mismas particiones, para que la
comparación sea justa:

| Modelo | Tipo | Conocimiento previo |
|---|---|---|
| **TF-IDF + Softmax** | Línea base clásica (regresión softmax en PyTorch) | Ninguno |
| **BiLSTM + atención** | Red recurrente entrenada **desde cero** | Ninguno |
| **NewsHybridNet** | Arquitectura propia de doble rama con fusión por compuerta | Léxico (embeddings fastText congelados) |
| **DistilBERT** | Transformer preentrenado, ajustado (fine-tuning) | Contextual profundo |

Además de las métricas (macro F1, accuracy, latencia, brecha de generalización), el proyecto incluye
una **interfaz web** para probar los modelos de forma interactiva.

El diseño completo (problema, datos, arquitecturas, hiperparámetros, evaluación) está en
[`Anteproyecto.md`](Anteproyecto.md).

---

## Estructura del repositorio

```
ProyectoFinal-DeepLearning/
├── Anteproyecto.md                 # Documento de diseño del proyecto
├── README.md                       # Este archivo
├── models/
│   ├── train_models.ipynb          # Notebook: entrena y evalúa los 4 modelos
│   └── results/checkpoints/        # Pesos entrenados (salida del notebook)
└── app/                            # Interfaz web
    ├── backend/                    # API FastAPI + carga de modelos
    ├── frontend/                   # HTML/CSS/JS propios
    └── README.md                   # Detalles de la interfaz
```

---

## Requisitos

- **Python 3.10+**
- Para entrenar: GPU recomendada (Kaggle/Colab). El notebook también corre en CPU, más lento.
- Para la interfaz: basta CPU.

---

## Parte 1 — Entrenar los modelos

El notebook `models/train_models.ipynb` está pensado para **Kaggle** (con fallback local) y
**guarda un checkpoint tras cada época**, de modo que si la sesión se corta, al reejecutar **se reanuda
automáticamente** desde la última época.

### En Kaggle (recomendado)

1. Sube `models/train_models.ipynb` como notebook de Kaggle.
2. Activa **Settings → Internet: On** (para descargar AG News y DistilBERT).
3. (Opcional, para NewsHybridNet con fastText real) añade un dataset de fastText inglés `.bin`.
4. Ejecuta todas las celdas. Los pesos quedan en `/kaggle/working/checkpoints/`.
5. Descarga la carpeta `checkpoints/` y cópiala a `models/results/checkpoints/` en este repo.

### En local

```bash
pip install -U torch "transformers>=4.40" "datasets>=2.19" "scikit-learn>=1.3" scipy joblib gensim jupyter
jupyter notebook models/train_models.ipynb
```

### Qué guarda cada modelo (en `models/results/checkpoints/<modelo>/`)

- `last.pt` — última época (modelo, optimizador, scheduler, métricas, estados RNG); escritura atómica.
- `best.pt` — mejor macro F1 de validación (el que usa la interfaz).
- `history.json` — curvas de loss y métricas por época (incluye `epoch_seconds`).
- Artefactos para servir: vocabulario, vectorizador TF-IDF, tokenizer, embeddings, config de n-gramas.

### Qué produce el notebook de entrenamiento (en `OUTPUT_DIR`)

Para cubrir los lineamientos del profesor, además de los checkpoints se generan:

- `curvas_aprendizaje.png` — curvas de loss y métricas (train/val) de los 4 modelos (sección 8.1).
- `recursos_computacionales.json` — hardware (CPU/GPU/RAM) y tiempo de entrenamiento por modelo (8.2).
- `ablaciones_newshybridnet.csv` — comparación de las ablaciones de NewsHybridNet (sección 10).
- `multisemilla_newshybridnet.csv` — resultados por semilla y media ± desviación (sección 11).
- `resumen_resultados.csv` — accuracy y macro F1 en test de los 4 modelos.

### Evaluación sin reentrenar (`models/evaluate_models.ipynb`)

Notebook aparte que **solo carga los `best.pt`** (no reentrena) y produce: matriz de confusión y
ejemplos mal clasificados, protocolo de latencia, evaluación en el conjunto externo BBC con la brecha
de generalización, prueba de McNemar entre modelos y la distribución de la compuerta de NewsHybridNet.
Pensado para reejecutarse muchas veces mientras se prepara el informe.

---

## Parte 2 — Ejecutar la interfaz web

La interfaz (FastAPI + frontend propio) deja pegar un titular o cuerpo de noticia **en inglés** y
compara en paralelo la predicción de los cuatro modelos, con barras de probabilidad y, para
NewsHybridNet, el mapa de atención y el peso de la compuerta semántica/léxica.

```bash
cd app/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Abre **http://localhost:8000**

### Modo demo vs. modelos reales

- Si **no** existen los checkpoints en `models/results/checkpoints/`, la app funciona en **modo demo**:
  predicciones simuladas deterministas, marcadas con una insignia `demo`. Útil para ver/desarrollar la
  interfaz antes de tener los pesos.
- En cuanto se copian los checkpoints reales a `models/results/checkpoints/`, la app los detecta y usa
  **automáticamente** (insignia `real`), sin tocar código.

### Conectar los modelos entrenados

Copia la carpeta `checkpoints/` generada por el notebook de modo que quede así:

```
models/results/checkpoints/
├── tfidf_softmax/     best.pt, tfidf_vectorizer.joblib
├── rnn_scratch/       best.pt, vocab.json
├── newshybridnet/     best.pt, vocab.json, embeddings.npy, ngram_config.json
└── distilbert/        hf_best/ (carpeta), tokenizer/ (carpeta)
```

Reinicia el servidor y la insignia de cada modelo pasará de `demo` a `real`.

---

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/health` | Estado del servicio y lista de clases. |
| `GET` | `/api/models` | Metadatos de los 4 modelos y si tienen pesos disponibles. |
| `POST` | `/api/predict` | `{ "text": "...", "model": "<key opcional>" }` → predicciones. |

Ejemplo:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Wall Street stocks tumble as oil prices surge."}'
```

---

## Datos

- **Dataset:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (`fancyzhx/ag_news`),
  120.000 ejemplos de entrenamiento y 7.600 de prueba, 4 clases balanceadas, en inglés.
- **Particiones:** 10 % estratificado del train como validación (semilla fija) → ~108.000 train /
  ~12.000 val. El test oficial solo se usa al final. El vocabulario se construye solo con el train.
- **Generalización (opcional):** conjunto externo (p. ej. BBC News) para medir la caída de desempeño
  fuera de distribución.

---

## Créditos

Proyecto académico. Dataset AG News vía Hugging Face `datasets`. DistilBERT vía Hugging Face
`transformers`. Embeddings fastText (Facebook AI Research).
