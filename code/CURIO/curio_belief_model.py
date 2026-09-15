import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    LLM-based encoder for conversation text.
    Uses a frozen HuggingFace transformer (e.g., MiniLM, BERT, RoBERTa).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

        # Encoder is frozen by default; unfreeze intentionally if needed by setting
        # param.requires_grad = True and adding encoder params to the optimizer.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, T)
        mask: (B, T)
        returns: (B, H) pooled representation
        """
        outputs = self.model(input_ids=ids, attention_mask=mask)
        token_emb = outputs.last_hidden_state

        mask_expanded = mask.unsqueeze(-1).float()
        masked_emb = token_emb * mask_expanded
        summed = masked_emb.sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        return summed / counts


class CurioBeliefModel(nn.Module):
    """
    CURIO belief model with a primary archetype head and an auxiliary
    latent-trait regression head sharing the same text encoder.
    """
    def __init__(
        self,
        num_types: int,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        temp: float = 0.3
    ):
        super().__init__()
        self.temp = temp
        self.encoder = TextEncoder(model_name=model_name)
        
        # archetype head
        self.type_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(128, num_types)
        )

        # latent trait head
        self.latent_head = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 8),
            nn.Sigmoid(),
        )

    def forward(self, ids, mask):
        embeddings = self.encoder(ids, mask)

        logits = self.type_head(embeddings)
        
        # temperature scaling
        # temp > 1 makes the distribution flatter (more uncertain)
        # temp < 1 makes it "peakier" (more certain)
        probs = torch.softmax(logits / self.temp, dim=-1)
        latent= self.latent_head(embeddings)
        return logits, probs, latent


if __name__ == "__main__":
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_types = 10
    model = CurioBeliefModel(num_types, model_name)

    # Example batch of text
    texts = [
        "I’m looking for something affordable and durable.",
        "I care most about premium brands and high-end materials.",
    ]

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=64,
    )

    logits, belief, latent = model(batch["input_ids"], batch["attention_mask"])
    print(f"Logits: {logits.shape}")
    print(f"Belief: {belief.shape}")
    print(f"Latent: {latent.shape}")
