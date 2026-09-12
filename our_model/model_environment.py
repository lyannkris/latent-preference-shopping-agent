import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from typing import List, Dict, Tuple
import random

from model_belief import BeliefEncoder
from reward_module import total_reward, intrinsic_reward_gaussian
from product_scoring import recommend_product

class ModelEnvironment:
    def __init__(self,
                 belief_model: BeliefEncoder,
                 vocab_size: int,
                 product_embeddings,
                 tokenizer,
                 device,
                 latent_dim: int = 8,
                 max_turns: int = 5,
                 confidence_threshold: float = 0.15
                 ):
        self.belief_model = belief_model
        self.device=device
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_turns = max_turns
        self.confidence_threshold = confidence_threshold
        self.tokenizer = tokenizer
        self.history = []
        self.turn = 0
        self.user = None
        self.user_pref = None 
        self.product_embeddings = product_embeddings

    def _encode_history(self):
        # Join the ENTIRE conversation history
        dialogue = "".join(f"{msg['role']}: {msg['content']}\n" for msg in self.history)
        tokens = self.tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=512)
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

    def _run_belief(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_samples: int = 5):
        self.belief_model.train() 
        logit_samples = []
        for _ in range(num_samples):
            logits = self.belief_model(input_ids, attention_mask)
            logit_samples.append(logits)
        
        stacked_logits = torch.stack(logit_samples)
        
        # Apply sigmoid first
        stacked_probs = torch.sigmoid(stacked_logits)
        
        # Now calculate mu and std on the 0-1 values
        final_mu = stacked_probs.mean(dim=0) 
        final_std = stacked_probs.std(dim=0) + 1e-4
        
        return final_mu, final_std

    def reset(self):
        self.history = []
        self.turn = 0

        # --- THE MC DROPOUT NATURAL PRIOR ---
        # Pass an empty interaction to the network so it outputs
        # the true population baseline for mu and std using the 5 passes.
        tokens = self.tokenizer("", return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        
        with torch.no_grad():
            base_mu, base_std = self._run_belief(input_ids, attention_mask, num_samples=5)
        
        # Save the detached true baseline as our starting state
        self.current_mu = base_mu.detach()
        self.current_std = base_std.detach()

        return {
            "history": self.history,
            "mu": self.current_mu.clone(),
            "std": self.current_std.clone()
        }

    # Removed turn_mask argument
    def step(self, agent_message: str, alpha: float = 1.0, gamma: float = 1.0):
        # Grab belief state BEFORE the user replies
        mu_before = self.current_mu.clone()
        std_before = self.current_std.clone()

        # Append the Agent's message
        self.history.append({"role": "assistant", "content": agent_message})

        # User reply
        user_message = self.user.converse_with_user(agent_message, use_llm=False)
        self.history.append({"role": "user", "content": user_message})

        # Run Belief Encoder on the new history
        input_ids, attention_mask = self._encode_history()
        
        # Run MC Dropout
        mu_predicted, std_predicted = self._run_belief(input_ids, attention_mask, num_samples=5)
        std_predicted = std_predicted.detach()

        # REMOVED ARCHITECTURAL GATING
        # The model now updates all 8 dimensions simultaneously based on the text
        mu_after = mu_predicted
        std_after = std_predicted

        # Save detached state for the next turn
        self.current_mu = mu_after.detach()
        self.current_std = std_after.detach()

        # Turn update
        self.turn += 1
        done = (self.turn >= self.max_turns) 

        if done:
            self.done = True 
            best_asin, _ = recommend_product(mu_after, self.product_embeddings)

            # Removed turn_mask
            rewards = total_reward(
                chosen_asin=best_asin,
                item_features=self.product_embeddings,
                user_pref=self.user.latent_variables,
                std_before=std_before,
                std_after=std_after,
                alpha=alpha,
                gamma=gamma,
            )

            obs = {
                "history": self.history,
                "mu": mu_after, 
                "std": std_after,
                "recommendation": best_asin,
            }

            return obs, rewards["total"], True, rewards
        
        # Compute intrinsic reward (Removed turn_mask)
        r_int = intrinsic_reward_gaussian(std_before, std_after)
        intermediate_reward = alpha * float(r_int.detach())

        # Calculate the RAW intrinsic reward (for math/metrics)
        raw_intrinsic = float(r_int.detach())

        obs = {
            "history": self.history,
            "mu": mu_after,
            "std": std_after
        }

        info = {
            "intrinsic_raw": raw_intrinsic,               # The pure info gain
            "intrinsic_scaled": intermediate_reward,      # What the model sees
            "user_message": user_message
        }

        return obs, intermediate_reward, False, info