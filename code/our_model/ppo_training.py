import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from typing import List
import random
import pandas as pd

import time
from datetime import datetime

from model_environment import ModelEnvironment

LATENT_KEYS = ["comfort", "functionality", "maintenance", "status_symbol",
                "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"]

answerable_attributes = [
                "age",
                "gender",
                "clothing size",
                "income",
                "location",
                "climate",
                "activity level",
                "social frequency",
                "world news awareness",
                "brand awareness",
                "highest education level",
                "name",
                "personality",
                "occupation",
                "relationship status",
                "languages spoken",
                "places traveled",
                "pet ownership",
                "sibling count",
                "life goals",
                "attending event"
            ]

attr_to_latent_map = {
    "age":          [0, 1, 0, 1, 0, 1, 1, 1], # Impacts almost everything in your logic
    "gender":       [0, 1, 0, 0, 0, 0, 1, 1],
    "income":       [0, 0, 0, 1, 0, 0, 0, 1], # Only status and trend
    "location":     [1, 0, 0, 1, 0, 0, 0, 1],
    "activity level": [1, 1, 0, 0, 1, 0, 0, 0],
    "social frequency": [0, 0, 0, 1, 0, 0, 0, 1],
    "world news awareness": [0, 0, 0, 0, 0, 1, 0, 0],
    "brand awareness": [0, 0, 0, 1, 0, 0, 0, 1],
    "highest education level": [0, 0, 0, 0, 0, 1, 0, 0],
    "personality":  [1, 0, 0, 0, 1, 0, 1, 1],
}

# -----------------------------
# VALUE NETWORK
# -----------------------------
class ValueNetwork(nn.Module):
    def __init__(self, latent_dim: int, questions_dim, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # --- FIX: Added LayerNorm to stabilize high-variance inputs ---
            nn.LayerNorm((2 * latent_dim) + questions_dim), 
            nn.Linear((2 * latent_dim) + questions_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mu: torch.Tensor, std: torch.Tensor, masks) -> torch.Tensor:
        x = torch.cat([mu, std, masks], dim=-1)
        return self.net(x)
# -----------------------------
# SIMPLE LLM ACTOR (placeholder for question generation)
# -----------------------------
class LLMActor(nn.Module):
    """
    Returns a dummy question string. Replace with real LLM API if needed.
    """
    def __init__(self, latent_keys: List[str], answerable_attributes: List[str] = None, hidden_dim: int = 64):
        super().__init__()
        
        self.latent_keys = latent_keys
        self.latent_dim = len(latent_keys)

        # self.client = OpenAI(
        #     base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
        #     api_key="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
        # )
        # self.model_name = "openai/gpt-oss-120b"

        if answerable_attributes is None:
            self.answerable_attributes = [
                "age",
                "gender",
                "clothing size",
                "income",
                "location",
                "climate",
                "activity level",
                "social frequency",
                "world news awareness",
                "brand awareness",
                "highest education level",
                "name",
                "personality",
                "occupation",
                "relationship status",
                "languages spoken",
                "places traveled",
                "pet ownership",
                "sibling count",
                "life goals",
                "attending event"
            ]
        else:
            self.answerable_attributes = answerable_attributes

        self.num_attrs = len(self.answerable_attributes)

        # Policy network: input = mu + std (2 * latent_dim), output = logits over attributes
        self.policy_net = nn.Sequential(
            nn.LayerNorm(2 * self.latent_dim + self.num_attrs),
            nn.Linear(2 * self.latent_dim + self.num_attrs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_attrs)
        )

    def forward(self, mu: torch.Tensor, std: torch.Tensor, action_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass: returns logits over attributes.
        mu, std: tensors of shape (batch_size, latent_dim)
        """
        if action_mask is not None:
            x = torch.cat([mu, std, action_mask], dim=-1)
        else:
            x = torch.cat([mu, std], dim=-1)
        logits = self.policy_net(x)
        
        if action_mask is not None:
            # action_mask is a binary tensor (1 means already asked)
            logits = logits.masked_fill(action_mask == 1, -1e9)
            
        return logits

    def generate_question(self, mu: torch.Tensor, std: torch.Tensor, current_mask: torch.Tensor):
        """
        Sample an attribute from the learned policy and generate a question.
        Returns: question, action, log_prob
        """
        # 1. Get logits using the mask!
        logits = self.forward(mu, std, action_mask=current_mask)

        # print(f"Logits: {logits.detach().cpu().numpy()}")

        # 2. Sample action
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        target_attr = self.answerable_attributes[action.item()]
        question = f"Could you tell me more about your {target_attr}?"

        return question, action, log_prob

# -----------------------------
# PPO helper functions
# -----------------------------
def compute_returns(rewards: list, gamma: float = 0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def ppo_update(value_net, actor, optimizer,
               zs, stds, actions, log_probs_old, returns, masks, episode,
               clip_eps=0.2, epochs=4):

    zs_t = torch.cat(zs, dim=0).detach()
    stds_t = torch.cat(stds, dim=0).detach()
    masks_t = torch.cat(masks, dim=0).detach()

    # actions_t = torch.stack(actions).long()
    # old_log_probs_t = torch.stack(log_probs_old).detach()
    # NEW:
    # Use .cat() to flatten the list of [1] tensors into a clean 1D [N] tensor
    actions_t = torch.cat(actions).long()
    old_log_probs_t = torch.cat(log_probs_old).detach()

    returns_t = torch.tensor(
        [r.detach().item() if torch.is_tensor(r) else r for r in returns],
        dtype=torch.float32,
        device=zs_t.device
    ).unsqueeze(-1)

    # =========================================================
    # 1. CALCULATE FIXED ADVANTAGES (OUTSIDE THE LOOP)
    # =========================================================
    # We use torch.no_grad() because we just want the BATCH'S initial opinion,
    # we don't want to accidentally calculate gradients here.
    with torch.no_grad():
        initial_values = value_net(zs_t, stds_t, masks_t)
        
    advantages = returns_t - initial_values
    
    # Normalize the fixed advantages
    if advantages.shape[0] > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Accumulators for this specific update
    update_policy_losses = []
    update_value_losses = []
    update_entropies = []

    for _ in range(epochs):
        # 1. Forward passes
        logits = actor.forward(zs_t, stds_t, action_mask=masks_t)
        values = value_net(zs_t, stds_t, masks_t)
        
        # 2. Policy math
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs_t)

        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
        if _ == 0:
            print(f"Clip Fraction (Epoch 1): {clip_frac:.4f}")


        # 4. Surrogate losses
        surr1 = ratio * advantages.squeeze()
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.squeeze()

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns_t)

        # 5. Combined Loss (Value coefficient of 1.0 for now)

        # total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        # total_loss = policy_loss + 0.5 * value_loss
        entropy_coef = max(0.001, 0.05 * (0.995 ** episode)) 
        total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        # 6. Step
        optimizer.zero_grad()
        total_loss.backward()

        # --- NEW: GRADIENT CLIPPING ---
        # This clamps the total L2 norm of the gradients to a maximum of 0.5.
        # It preserves the direction, but scales down the magnitude.
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)

        optimizer.step()

        if _ == 0: 
            # Grab the first linear layer of the Actor
            first_layer = actor.policy_net[1] # Assuming [0] is LayerNorm, [1] is Linear
            
            if first_layer.weight.grad is None:
                print("🚨 CRITICAL: NO GRADIENTS AT ALL. The Actor is disconnected from the Loss!")
            else:
                grad_mean = first_layer.weight.grad.abs().mean().item()
                print(f"Actor Gradient Magnitude: {grad_mean:.6f}")

        # 7. Record metrics
        update_policy_losses.append(policy_loss.item())
        update_value_losses.append(value_loss.item())
        update_entropies.append(entropy.item())

        # if _ == 0: # Only print once per update
        #     print(f"Advantage Mean: {advantages.mean().item():.4f} | Max: {advantages.max().item():.4f}")

    # Return the average of each metric for this batch update
    return {
        "policy_loss": sum(update_policy_losses) / len(update_policy_losses),
        "value_loss": sum(update_value_losses) / len(update_value_losses),
        "entropy": sum(update_entropies) / len(update_entropies)
    }


def sample_z(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(std)
    return mu + eps * std  # z ~ N(mu, diag(std^2))

# -----------------------------
# TRAINING LOOP
# -----------------------------
# 1. REMOVE SENSITIVITY_MATRIX from the function definition
def train_model_multiuser(env: ModelEnvironment,
                          value_net: ValueNetwork,
                          actor: LLMActor,
                          encoder_optimizer: torch.optim.Optimizer,
                          ppo_optimizer: torch.optim.Optimizer,
                          all_users: list,
                          num_episodes: int = 500,
                          alpha: float = 1.0,
                          gamma: float = 0.99):
    """
    Train CURIO+VPL across multiple simulated users.
    """

    training_log = []
    belief_log = [] # for z values

    start_time = time.time()

    UPDATE_FREQ = 25

    batch_zs, batch_stds, batch_actions = [], [], []
    batch_log_probs, batch_returns, batch_masks = [], [], []

    history = []
    ppo_metric_history = []

    for episode in range(num_episodes):
        episode_mses = []

        chosen_user = random.choice(all_users)
        chosen_user.reset_convo()

        env.user = chosen_user
        env.user_pref = chosen_user.latent_variables
        
        obs = env.reset()
        zs, stds, rewards, actions, log_probs, masks = [], [], [], [], [], [] # ADDED MASKS LIST

        # --- INITIALIZE MASK & ACTION SPACE DROPOUT ---
        current_mask = torch.zeros(1, actor.num_attrs, device=obs["mu"].device)
        # num_to_mask = random.randint(5, 10) # Drop a few random attributes from possible asking list at turn 0
        # disabled_indices = random.sample(range(actor.num_attrs), num_to_mask)
        # for idx in disabled_indices:
        #     current_mask[0, idx] = 1
        true_latent = torch.tensor(
            [chosen_user.latent_variables[k] for k in LATENT_KEYS],
            dtype=torch.float32,
            device=obs["mu"].device
        ).unsqueeze(0)

        cumulative_encoder_loss = 0.0 # Track NLL across the episode

        ep_intrinsic_reward = 0.0
        ep_extrinsic_reward = 0.0

        done = False
        turn = 0
        while not done:
            mu, std = obs["mu"], obs["std"]
            stds.append(std)
            zs.append(mu)

            if turn == 0:
                turn0_std_captured_at_reset = float(std.mean().item())
            
            # 1. Generate question
            question, action, log_prob = actor.generate_question(mu, std, current_mask)
            masks.append(current_mask.clone())
            current_mask[0, action.item()] = 1

            # 2. Step the environment (NO MASK PASSED)
            obs, reward, done, info = env.step(question, alpha=alpha, gamma=gamma)

            if "intrinsic_scaled" in info:
                ep_intrinsic_reward += (float(info["intrinsic_scaled"]))
                
            if "extrinsic" in info:
                ep_extrinsic_reward += float(info["extrinsic"])

            # ... [keep your belief_log appending] ...
            chosen_attr_for_question = question.replace("Could you tell me more about your ", "").replace("?", "").strip()

            belief_log.append({
                "user_pref": chosen_user.latent_variables,
                "episode": episode,
                "turn": turn,
                "chosen_attr_for_question": chosen_attr_for_question,
                "mu": mu.detach().cpu().numpy().tolist(),
                "std": std.detach().cpu().numpy().tolist(),
                "std_mean": float(std.mean().item()),
                "std_per_dim": std.detach().cpu().numpy().squeeze().tolist()
            })
            turn += 1

            print(f"Initial Mu: {mu}")
            print(f"Initial Std: {std}")
            print(f"Model: {question}")
            print(f"User Response: {info.get('user_message')}")
            print(f"New Mu: {obs['mu']}")
            print(f"New Std: {obs['std']}")

            # --- FULL VECTOR MSE LOSS ---
            new_mu = obs["mu"]
            
            # 1. Calculate Full Error for Logging & Loss
            step_mse = ((new_mu - true_latent) ** 2)
            episode_mses.append(step_mse.mean())

            # 2. Backpropagate the Full Vector directly
            # No more anchoring masks; the model must learn to keep
            # unrelated variables stable on its own.
            cumulative_encoder_loss += step_mse.mean()
            # -------------------------------------------

            rewards.append(reward)
            actions.append(action)
            log_probs.append(log_prob)

        returns = compute_returns(rewards, gamma)

        # 1. Backpropagate the Encoder EVERY episode 
        # (Supervised learning handles small batches just fine)
        final_encoder_loss = (cumulative_encoder_loss / len(rewards)) * 0.1 
        final_encoder_loss = final_encoder_loss.sum()
        
        encoder_optimizer.zero_grad()
        final_encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(belief_model.parameters(), max_norm=1.0)
        encoder_optimizer.step()

        # 2. Add the episode's data to the PPO batch
        batch_zs.extend(zs)
        batch_stds.extend(stds)
        batch_actions.extend(actions)
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
        batch_masks.extend(masks)

        # 3. ONLY run PPO Update when the batch is full
        if (episode + 1) % UPDATE_FREQ == 0:
            metrics = ppo_update(
                value_net, actor, ppo_optimizer, 
                batch_zs, batch_stds, batch_actions, 
                batch_log_probs, batch_returns, batch_masks, episode, epochs=8
            )

            ppo_metric_history.append({
                "episode": episode,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"]
            })

            with open("logs/ppo_metrics.json", "w") as f:
                json.dump(ppo_metric_history, f, indent=4)
            
            # Clear the batches after updating
            batch_zs, batch_stds, batch_actions = [], [], []
            batch_log_probs, batch_returns, batch_masks = [], [], []

        total_r = sum(r.detach().item() if torch.is_tensor(r) else float(r) for r in rewards)

        training_log.append({
            "episode": episode,
            "total_reward": total_r,
            "num_turns": len(rewards)
        })

        history.append({
            "episode": episode,
            "mse_start": episode_mses[0].item(),   # Knowledge at Turn 0
            "mse_end": episode_mses[-1].item(),    # Knowledge at Turn 10
            "mse_avg": (sum(episode_mses) / len(episode_mses)).item(), # Overall Trend
            "turn0_var": turn0_std_captured_at_reset,
            "extrinsic_reward": ep_extrinsic_reward,   # <--- ADDED
            "intrinsic_reward": ep_intrinsic_reward,   # <--- ADDED
            "total_reward": total_r
        })

        print(f"Episode {episode}/{num_episodes} - Total reward: {sum(rewards):.3f}")

        if episode % 10 == 0:
            elapsed = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(
                f"[{timestamp}] Episode {episode} | "
                f"Reward: {sum(rewards):.3f} | "
                f"Elapsed: {elapsed/60:.2f} min"
            )

            os.makedirs("logs", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)

            with open("logs/training_log.txt", "a") as f:
                f.write(f"Episode {episode} - Total reward: {sum(rewards):.3f}")
            
            with open("logs/history.json", "w") as f:
                json.dump(history, f, indent=2)

        if episode % 100 == 0:
            torch.save(value_net.state_dict(), "checkpoints/value_net_checkpoint.pth")
            #Added to save belief model
            torch.save(env.belief_model.state_dict(), "checkpoints/belief_model_checkpoint.pth")
            torch.save(actor.policy_net.state_dict(), "checkpoints/actor_policy_checkpoint.pth")
            print(f"Saved all checkpoints at episode {episode}")
            
        if episode % 50 == 0:
            with open("logs/training_metrics.json", "w") as f:
                json.dump(training_log, f, indent=2)
            with open("logs/belief_log.json", "w") as f:
                json.dump(belief_log, f, indent=2)


if __name__ == "__main__":
    from model_belief import BeliefEncoder
    from model_environment import ModelEnvironment
    from transformers import AutoTokenizer
    from utils.generate_users import Simulated_User

    with open("data/product_embeddings.json", "r") as f:
        product_embeddings = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained("./local_minilm")
    # -----------------------------
    # 1. Parameters
    # -----------------------------
    LATENT_DIM = 8           # number of latent traits
    NUM_USERS = 2000         # number of simulated users
    MAX_TURNS = 10
    NUM_EPISODES = 4000       # total training episodes
    ALPHA = 1.0             # weight for intrinsic reward
    GAMMA = 0.99             # discount factor
    ENCODER_LR = 5e-4 

    CRITIC_LR = 5e-3  # Faster: Needs to keep up with the shifting Encoder
    ACTOR_LR = 5e-4   # Slower: Prevents the policy from collapsing/spamming questions

    # -----------------------------
    # 2. Generate 2000 simulated users
    # -----------------------------
    all_users = pd.read_pickle("data/user_train_set.pkl")
    # all_users: List[torch.Tensor] = []
    # for _ in range(NUM_USERS):
    #     # Each user has a latent preference vector between 0 and 1
    #     user_pref = torch.rand(1, LATENT_DIM)
    #     all_users.append(user_pref)

    # -----------------------------
    # 3. Initialize models
    # -----------------------------
    belief_model = BeliefEncoder()
    value_net = ValueNetwork(latent_dim=LATENT_DIM, questions_dim=len(answerable_attributes))
    actor = LLMActor(latent_keys=LATENT_KEYS)

    belief_model.to(device)
    value_net.to(device)
    actor.policy_net.to(device)
    # Convert this to a Tensor for the training loop
    # Shape: [num_attributes, 8]

    encoder_optimizer = optim.Adam(belief_model.parameters(), lr=ENCODER_LR)

    ppo_optimizer = optim.Adam([
        {'params': value_net.parameters(), 'lr': CRITIC_LR},
        {'params': actor.policy_net.parameters(), 'lr': ACTOR_LR}
    ])

    # -----------------------------
    # 4. Initialize environment
    # -----------------------------
    env = ModelEnvironment(
        belief_model=belief_model,
        tokenizer=tokenizer,
        device=device,
        vocab_size=tokenizer.vocab_size,
        product_embeddings=product_embeddings,
        latent_dim=LATENT_DIM,
        max_turns=MAX_TURNS
    )

    # -----------------------------
    # 5. Train
    # -----------------------------
    train_model_multiuser(
        env=env,
        value_net=value_net,
        actor=actor,
        encoder_optimizer=encoder_optimizer,
        ppo_optimizer=ppo_optimizer,
        all_users=all_users,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA
    )

    # -----------------------------
    # 6. Save models
    # -----------------------------
    os.makedirs("final_model", exist_ok=True)

    torch.save(belief_model.state_dict(), "final_model/belief_model_final.pth")
    torch.save(value_net.state_dict(), "final_model/value_net_final.pth")
    torch.save(actor.policy_net.state_dict(), "final_model/actor_policy_net.pth")

    # Save actor config
    actor_config = {
        "latent_dim": LATENT_DIM,
        "answerable_attributes": actor.answerable_attributes,
        "hidden_dim": 64  # matches the hidden_dim used in LLMActor
    }
    with open("final_model/actor_config.json", "w") as f:
        json.dump(actor_config, f, indent=2)

    print("Training finished. Models and actor config saved in ./final_model/")