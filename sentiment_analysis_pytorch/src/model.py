import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", n_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use pooled output if available, otherwise mean pool
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            pooled = last_hidden.mean(dim=1)
        x = self.dropout(pooled)
        return self.classifier(x)