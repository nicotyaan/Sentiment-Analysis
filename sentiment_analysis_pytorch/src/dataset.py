import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, model_name="bert-base-uncased", max_length=256):
        # Use the slow tokenizer to avoid additional dependencies like sentencepiece
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item