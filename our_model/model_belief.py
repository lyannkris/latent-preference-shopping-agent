import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from typing import Dict, List, Tuple

class BeliefEncoder(nn.Module):
    """
    Encodes user interaction history into a Gaussian posterior over latent vector z.
    We are using a pretrained encoder so that we can focus on just the encoded sentences
    rather than having to also train an encoder as well.
    """
    # downloaded the files locally into a folder /local_minilm
    def __init__(self, model_name="./local_minilm", latent_dim=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_dim = self.encoder.config.hidden_size
        self.latent_dim = latent_dim
        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.3) # Try 0.3 or 0.4 here

        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                self.dropout,             # <-- Scramble the base 768-dim embedding
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(negative_slope=0.01), # Prevents dead neurons
                self.dropout,             # <-- Scramble the 64-dim hidden layer
                nn.Linear(64, 1)
            ) for _ in range(latent_dim)
        ])

        # --- FORCE HIGH INITIAL VARIANCE ---
        # Initialize the linear layers with massive weights so MC Dropout
        # creates huge swings before the network is fully trained.
        for m in self.mu_heads.modules():
            if isinstance(m, nn.Linear):
                # PyTorch default std is ~0.03. We blow it up to 0.5.
                nn.init.normal_(m.weight, mean=0.0, std=0.1) 
                nn.init.constant_(m.bias, 0.0)
        # ----------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device) 
        attention_mask = attention_mask.to(device)
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:, 0, :] 
        h = self.pre_head_norm(h) # Stabilizes the input magnitudes

        # Removed torch.sigmoid() from this line!
        logits = torch.cat([head(h) for head in self.mu_heads], dim=-1)
        
        return logits
