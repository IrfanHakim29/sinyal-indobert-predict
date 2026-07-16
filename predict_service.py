"""
Live IndoBERT prediction service for the "Coba Sendiri" feature.

Runs on the dynamically-quantized ONNX export (pipeline/export_onnx.py) via raw
onnxruntime + tokenizers — deliberately NOT transformers/torch, which alone add
~250-450MB of import overhead. That's the difference between fitting in a free
512MB host and not. Same code path locally and in production, so what you test
here is exactly what's deployed.

Aspect detection reuses the exact lexicon + Sastrawi stemming from the data
pipeline so live results are consistent with the dataset.

Run:  python serve/predict_service.py     (listens on 127.0.0.1:8000)
"""

import os
import re
import urllib.request

import emoji
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tokenizers import Tokenizer

MAX_LEN = 128
NAMES = ["positive", "negative", "neutral"]  # label2id: positive:0, negative:1, neutral:2
MODEL_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.txt", "special_tokens_map.json", "model_quantized.onnx",
]
# Vercel's Python function bundler doesn't reliably ship large/git-lfs sibling
# folders (confirmed: NO_SUCHFILE at runtime even though the folder is committed
# and un-ignored) — so on a fresh cold start we pull the files straight from
# Blob storage into /tmp instead of depending on them being bundled.
BLOB_BASE = os.environ.get(
    "MODEL_BLOB_BASE_URL", "https://lm9jkddarba1yxc5.public.blob.vercel-storage.com"
)

ASPECT_LEXICON = {
    "pengiriman": {"kirim", "paket", "kurir", "antar", "ekspedisi", "sampai", "tiba", "cod"},
    "kemasan": {"kemas", "packing", "bungkus", "segel", "plastik", "dus"},
    "kualitas_produk": {"bagus", "rusak", "pecah", "cacat", "awet", "original", "ori",
                        "busuk", "segar", "kualitas", "mutu", "bocor"},
    "harga": {"harga", "murah", "mahal", "worth", "promo", "diskon", "ongkir"},
    "pelayanan": {"layan", "respon", "ramah", "balas", "komplain", "tanggap"},
}


def resolve_model_dir() -> str:
    """Local dev: model already sits on disk next to this script or at the
    project root. Deployed: download once from Blob into /tmp (warm
    invocations on the same instance reuse the cached copy)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "best-indobert-sentiment-onnx"),
        os.path.join(os.path.dirname(here), "best-indobert-sentiment-onnx"),
    ):
        if os.path.exists(os.path.join(candidate, "model_quantized.onnx")):
            return candidate

    cache_dir = "/tmp/best-indobert-sentiment-onnx"
    os.makedirs(cache_dir, exist_ok=True)
    for fname in MODEL_FILES:
        dest = os.path.join(cache_dir, fname)
        if not os.path.exists(dest):
            print(f"[predict] downloading {fname} from blob …", flush=True)
            urllib.request.urlretrieve(f"{BLOB_BASE}/{fname}", dest)
    return cache_dir


ONNX_DIR = resolve_model_dir()

print("[predict] loading quantized ONNX model …", flush=True)
session = ort.InferenceSession(
    os.path.join(ONNX_DIR, "model_quantized.onnx"), providers=["CPUExecutionProvider"]
)
tokenizer = Tokenizer.from_file(os.path.join(ONNX_DIR, "tokenizer.json"))
tokenizer.enable_truncation(max_length=MAX_LEN)
_input_names = {i.name for i in session.get_inputs()}
stemmer = StemmerFactory().create_stemmer()
print("[predict] ready.", flush=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    return re.sub(r"\s+", " ", text).strip()


def detect_aspects(text: str):
    tokens = set(stemmer.stem(clean_text(text)).split())
    return [a for a, kws in ASPECT_LEXICON.items() if tokens & kws]


def run_model(text: str):
    enc = tokenizer.encode(text)
    n = len(enc.ids)
    inputs = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.zeros((1, n), dtype=np.int64),
    }
    inputs = {k: v for k, v in inputs.items() if k in _input_names}
    logits = session.run(None, inputs)[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return probs


class Req(BaseModel):
    text: str


app = FastAPI(title="SINYAL — IndoBERT predict")
# CORS_ORIGINS env var: comma-separated allowed origins (e.g. the deployed Vercel URL).
# Falls back to "*" for demo hosting where the frontend origin isn't fixed yet.
_origins = os.environ.get("CORS_ORIGINS", "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",")] if _origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu", "quantized": True}


@app.post("/predict")
def predict(req: Req):
    text = (req.text or "").strip()
    if not text:
        return {"error": "empty"}
    clean = clean_text(text)
    probs = run_model(clean)
    idx = int(probs.argmax())
    return {
        "sentiment": NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probs": {NAMES[i]: round(float(probs[i]), 4) for i in range(3)},
        "aspects": detect_aspects(text),
        "device": "cpu",
    }


if __name__ == "__main__":
    import uvicorn

    # Render (and most PaaS) inject PORT and expect a 0.0.0.0 bind.
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1" if "PORT" not in os.environ else "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
