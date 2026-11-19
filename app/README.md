# Interfaz del clasificador de noticias

Interfaz web (FastAPI + frontend propio) para clasificar noticias de AG News con
los cuatro modelos del proyecto: **TF-IDF + Softmax**, **BiLSTM + atención**,
**NewsHybridNet** y **DistilBERT**.

## Estructura

```
app/
├── backend/
│   ├── main.py            # API FastAPI + sirve el frontend
│   ├── registry.py        # carga perezosa de los 4 modelos (con modo demo)
│   ├── models_def.py      # arquitecturas PyTorch (idénticas al notebook)
│   ├── preprocessing.py   # tokenización y n-gramas (idénticos al notebook)
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

Los pesos se leen de `../models/checkpoints/<modelo>/` (los genera el notebook
`models/train_models_kaggle.ipynb`).

## Modo demo (sin modelos entrenados)

Si todavía no existen los checkpoints, cada modelo responde en **modo demo**: una
predicción simulada y determinista basada en palabras clave. La interfaz funciona
igual y lo indica con una insignia `demo`. En cuanto se copien los checkpoints
reales a `models/checkpoints/`, la app los detecta y usa **automáticamente**
(campo `demo: false`), sin tocar código.

## Ejecutar

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Abrir http://localhost:8000

> El modo demo no necesita `torch` ni `transformers`; para servir modelos reales
> sí hacen falta (ya están en `requirements.txt`).

## Checkpoints esperados por modelo

| Modelo (`key`) | Carpeta | Archivos requeridos |
|---|---|---|
| TF-IDF + Softmax (`tfidf_softmax`) | `models/checkpoints/tfidf_softmax/` | `best.pt`, `tfidf_vectorizer.joblib` |
| BiLSTM + atención (`rnn_scratch`) | `models/checkpoints/rnn_scratch/` | `best.pt`, `vocab.json` |
| NewsHybridNet (`newshybridnet`) | `models/checkpoints/newshybridnet/` | `best.pt`, `vocab.json`, `embeddings.npy`, `ngram_config.json` |
| DistilBERT (`distilbert`) | `models/checkpoints/distilbert/` | `hf_best/`, `tokenizer/` |

## Nota sobre NewsHybridNet y el hashing

La rama léxica calcula los n-gramas de caracteres con `zlib.crc32`, que es determinista entre
procesos y máquinas (el `hash()` de Python no lo es salvo que se fije `PYTHONHASHSEED` antes de
arrancar el intérprete). El entrenamiento (`models/train_models.ipynb`), la evaluación
(`models/evaluate_models.ipynb`) y `preprocessing.py` usan la misma función, así que no hace falta
configurar nada. Los pesos entrenados con la versión anterior (`hash()`) no son compatibles: hay que
usar los de `models/checkpoints/` generados con esta versión.

## Endpoints

- `GET /api/health` — estado y lista de clases.
- `GET /api/models` — metadatos de los 4 modelos y si tienen pesos disponibles.
- `POST /api/predict` — `{ "text": "...", "model": "<key opcional>" }` → predicciones.
