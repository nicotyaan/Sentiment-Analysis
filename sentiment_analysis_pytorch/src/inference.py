import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.model import SentimentClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = BASE_DIR / "models" / "best_model.pt"

# If checkpoint is missing, fall back to a ready-to-use sentiment model.
# You can override it via SENTIMENT_FALLBACK_MODEL env var.
FALLBACK_MODEL_NAME = os.environ.get(
    "SENTIMENT_FALLBACK_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)

class RequestBody(BaseModel):
    text: str

app = FastAPI(title="Sentiment Analysis API")

# Load model and tokenizer once.
# - If `models/best_model.pt` exists, use it (fine-tuned checkpoint).
# - Otherwise, use a small pre-trained fallback model so the API still starts.
use_checkpoint = CHECKPOINT_PATH.exists()

if use_checkpoint:
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    model_name = ckpt.get("model_name", "bert-base-uncased")

    model = SentimentClassifier(model_name=model_name, n_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Keep it consistent with the dataset tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
else:
    model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_NAME).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME, use_fast=False)

    # SST-2: usually id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
    id2label = getattr(model.config, "id2label", {}) or {}

@app.post("/predict")
def predict(req: RequestBody):
    enc = tokenizer(
        req.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        if use_checkpoint:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).cpu().item())
        score = probs.cpu().numpy().tolist()[0]

    if use_checkpoint:
        label = "positive" if pred == 1 else "negative"
    else:
        raw = (id2label.get(pred) or "").upper()
        if raw:
            label = "positive" if "POS" in raw else "negative"
        else:
            label = "positive" if pred == 1 else "negative"
    return {"label": label, "score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)