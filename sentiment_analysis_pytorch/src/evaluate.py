import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.dataset import TextDataset
from src.model import SentimentClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "bert-base-uncased")
    model = SentimentClassifier(model_name=model_name, n_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, model_name

def evaluate(checkpoint_path, batch_size=32, max_length=256):
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ds = load_dataset("imdb")
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    model, device, model_name = load_model(checkpoint_path)
    test_dataset = TextDataset(test_texts, test_labels, model_name=model_name, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    preds = []
    trues = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, digits=4)
    cm = confusion_matrix(trues, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    return acc, report, cm

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py <checkpoint_path>")
        exit(1)
    evaluate(sys.argv[1])