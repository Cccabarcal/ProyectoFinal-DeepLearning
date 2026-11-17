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

## Nota importante sobre NewsHybridNet

La rama léxica usa `hash()` de Python para los n-gramas de caracteres, que **no es
estable entre procesos** salvo que se fije `PYTHONHASHSEED`. Para que la inferencia
reproduzca exactamente el entrenamiento, ejecutar el entrenamiento y el servidor
con el mismo valor, por ejemplo:

```bash
# Windows PowerShell
$env:PYTHONHASHSEED = "0"; uvicorn main:app --port 8000
```

Si el notebook se entrenó sin fijar el seed, lo ideal es sustituir `hash()` por una
función de hash determinista (p. ej. basada en `hashlib`) tanto en el notebook como
en `preprocessing.py`. Los demás modelos no se ven afectados.

## Endpoints

- `GET /api/health` — estado y lista de clases.
- `GET /api/models` — metadatos de los 4 modelos y si tienen pesos disponibles.
- `POST /api/predict` — `{ "text": "...", "model": "<key opcional>" }` → predicciones.
