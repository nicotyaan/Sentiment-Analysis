import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from src.dataset import TextDataset
from src.model import SentimentClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def train(
    model_name="distilbert-base-uncased",
    output_dir="models",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    max_length=256,
    device=None
):
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load IMDb dataset (uses the `datasets` library)
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    # small split for validation
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    train_dataset = TextDataset(tr_texts, tr_labels, model_name=model_name, max_length=max_length)
    val_dataset = TextDataset(val_texts, val_labels, model_name=model_name, max_length=max_length)
    test_dataset = TextDataset(test_texts, test_labels, model_name=model_name, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SentimentClassifier(model_name=model_name, n_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f} | Val acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name
            }, save_path)
            print(f"Saved best model to {save_path}")

    # final test evaluation
    print("Training finished. Best val acc:", best_val_acc)
    return os.path.join(output_dir, "best_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment classifier")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max token length")
    args = parser.parse_args()

    print(f"Using model: {args.model_name}")
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )