import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import pickle

from generate_users import Simulated_User
from curio_environment import CurioEnvironment
from transformers import AutoTokenizer
from curio_belief_model import CurioBeliefModel
from curio_environment import CurioEnvironment

from product_scoring import dict_to_vec, load_product_embeddings


LATENT_KEYS = [
    "comfort", "functionality", "maintenance", "status_symbol", 
    "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"
]


def load_fixed_users(path):
    with open(path, "rb") as f:
        users = pickle.load(f)
    print(f"Loaded users")
    return users


class CategoricalActor(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.latent_keys = [
            "comfort", "functionality", "maintenance", "status_symbol", 
            "modesty", "sustainability", "risk_tolerance", "trend_sensitivity"
        ]
        # The policy network: takes belief, outputs scores for each trait
        self.network = nn.Sequential( nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, len(self.latent_keys)))

    def generate_question(self, belief_vector: torch.Tensor, mask=None):
        """
        Returns question string
        """
        #Get Logits
        logits = self.network(belief_vector)
        if mask is not None:
          # Check if there is at least one valid action left
          if torch.all(mask == 0):
              # unmask everything if we run out of questions
              mask = mask.to(belief_vector.device)
              mask = torch.ones_like(mask) 
          logits = logits.masked_fill(mask == 0, -1e9)
        
        #Sample Action from distribution
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        # Map to Question
        target_attr = self.latent_keys[action_idx.item()]
        
        # Create question based on the attribute
        attr = target_attr.replace("_", " ")
        question = f"Could you tell me more about your {attr}?"
        
        return question, action_idx, log_prob



class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.net(belief)



def compute_returns(rewards, gamma: float = 0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def ppo_update(actor, value_net, optimizer, states, actions, action_masks, log_probs_old, returns, clip_eps=0.2):
    device = next(actor.parameters()).device 

    # Convert lists to tensors
    states_t = torch.stack(states).to(device).float()
    actions_t = torch.tensor(actions).to(device)
    action_masks_t = torch.stack(action_masks).to(device).float()
    old_log_probs_t = torch.stack(log_probs_old).to(device).float()
    returns_t = torch.tensor(returns).unsqueeze(-1).to(device).float()

    #Evaluate current policy
    logits = actor.network(states_t)
    logits = logits.masked_fill(action_masks_t == 0, -1e9)
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions_t)
    
    # Calculate Advantage
    values = value_net(states_t)
    advantages = (returns_t - values.detach()).squeeze(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    value_loss = nn.MSELoss()(values, returns_t)
 
    # clip objective
    ratio = torch.exp(new_log_probs - old_log_probs_t)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    dist_entropy = dist.entropy().mean()
    policy_loss = -torch.min(surr1, surr2).mean() # Add 0.01 entropy bonus
    value_loss = nn.MSELoss()(values, returns_t)

    total_loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy

    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    optimizer.step()


def train_curio( env: CurioEnvironment,
    value_net: ValueNetwork,
    belief_model,
    actor,
    optimizer: optim.Optimizer, belief_optimizer: optim.Optimizer,
    train_users=None,
    num_episodes: int = 2000,
    gamma: float = 0.99,
):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    belief_model.to(device)
    value_net.to(device)
    actor.to(device)
    
    criterion_belief = torch.nn.KLDivLoss(reduction="batchmean")
    criterion_latent = torch.nn.MSELoss()
    l_temp = 0.2
    latent_loss_weight = 0.0
    best_episode_reward = float("-inf")

    
    for episode in range(num_episodes):
   
        if train_users:
            user = train_users[episode % len(train_users)]
            user.reset_convo()
        else:
            user = Simulated_User()
        env.user = user
        obs = env.reset(new_user=False)
        true_type_idx = env.true_type_idx
        true_user_vec = dict_to_vec(user.get_latent_variables())

        type_distances = []
        for _, t_traits in env.user_types.items():
            t_vec = dict_to_vec(t_traits)
            type_distances.append(np.linalg.norm(true_user_vec - t_vec))
        soft_target = torch.softmax(
            -torch.tensor(type_distances, device=device, dtype=torch.float32) / l_temp,
            dim=0,
        )
        
        action_mask = torch.ones(len(LATENT_KEYS)).to(device)
        

        ep_beliefs = []
        ep_actions = []
        ep_action_masks = []
        ep_log_probs = []
        ep_rewards = []
        ep_loss_belief = []
        done = False

        while not done:
            # 1. Get input from environment
            input_ids, attention_mask = env._encode_history()
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            # 2. Get Belief 
            logits, type_probs, _ = belief_model(input_ids, attention_mask)

            # 3. Actor decision with action mask
            belief_input = type_probs.detach()
            policy_mask = action_mask.clone()

          
            agent_message, action_idx, lp = actor.generate_question(belief_input, mask=policy_mask)

            # Prevent asking same question
            action_mask[action_idx] = 0.0

            # ADD THIS after actor.generate_question:
            print(f"Turn {env.turn}: Agent Question -> {agent_message}")

            # 4. User answer
            obs, reward, done, info = env.step(agent_message)

            
            input_ids_new, attention_mask_new = env._encode_history()
            logits_new, _, latent_pred_new = belief_model(
                input_ids_new.to(device),
                attention_mask_new.to(device),
            )
          

            type_loss = criterion_belief( torch.log_softmax(logits_new, dim=-1),
                soft_target.unsqueeze(0),
            )

            latent_loss = criterion_latent(latent_pred_new, torch.tensor(true_user_vec, device=device, dtype=torch.float32).unsqueeze(0),
            )
            ep_loss_belief.append(type_loss + latent_loss_weight * latent_loss)

            reward = float(reward)

           
            ep_beliefs.append(belief_input.squeeze(0))
            ep_actions.append(int(action_idx.item()))
            ep_action_masks.append(policy_mask)
            ep_log_probs.append(lp.detach())
            ep_rewards.append(reward)
            
        if len(ep_loss_belief) > 0:
            belief_optimizer.zero_grad()
            # Sum loss
            total_belief_loss = torch.stack(ep_loss_belief).mean()
            total_belief_loss.backward()
            belief_optimizer.step()

        if len(ep_beliefs) > 0:
            returns = compute_returns(ep_rewards, gamma=gamma)
            returns = [float(r) for r in returns]
            ppo_update(actor,
                value_net,
                optimizer, ep_beliefs,
                ep_actions, ep_action_masks,
                ep_log_probs, returns,
            )

      
        if episode % 10 == 0: 
            print(f"Episode {episode}: total reward = {sum(ep_rewards):.3f}")
            with open("logs/training_log.txt", "a") as f:
                f.write(f"{episode},{sum(ep_rewards)}\n")

        episode_reward = float(sum(ep_rewards))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(value_net.state_dict(), "checkpoints/best_value_net_checkpoint.pth")
            torch.save(belief_model.state_dict(), "checkpoints/best_belief_model_checkpoint.pth")
            torch.save(actor.state_dict(), "checkpoints/best_actor_checkpoint.pth")
            with open("checkpoints/best_checkpoint_meta.json", "w") as f:
                json.dump(
                    {"episode": episode, "episode_reward": episode_reward,},
                    f,
                    indent=2,
                )
            
        #Save checkpoints
        if episode % 100 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(value_net.state_dict(), "value_net_checkpoint.pth")
            torch.save(belief_model.state_dict(), "belief_model_checkpoint.pth")
            torch.save(actor.state_dict(), "actor_checkpoint.pth")
            torch.save(value_net.state_dict(), f"checkpoints/value_net_ep{episode}.pth")
            torch.save(belief_model.state_dict(), f"checkpoints/belief_model_ep{episode}.pth")
            torch.save(actor.state_dict(), f"checkpoints/actor_ep{episode}.pth")
            print(f"Saved all checkpoints at episode {episode}")

if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from transformers import AutoTokenizer
    from generate_users import Simulated_User
    from product_scoring import load_product_embeddings, load_user_types

    import time
    from datetime import timedelta

    # Load data
    user_types = load_user_types("data/user_types.json")
    product_embeddings = load_product_embeddings("data/product_embeddings.json")
    num_user_types = len(user_types)

    # Belief model, tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    belief_model = CurioBeliefModel(
        num_types=8, 
        model_name=model_name,
    )

    value_net = ValueNetwork() 

    actor = CategoricalActor(input_dim=8)

    belief_model.train()
    
    belief_optimizer = torch.optim.Adam(
        list(belief_model.type_head.parameters()) + list(belief_model.latent_head.parameters()),
        lr=5e-3,
    )

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': 3e-4},
        {'params': value_net.parameters(), 'lr': 3e-4}
    ])

    env = CurioEnvironment( user=Simulated_User(), 
        user_types=user_types,
        product_map=product_embeddings,
        belief_model=belief_model,
        tokenizer=tokenizer,
        max_turns=8,
        alpha=2.0, gamma=1.0,     
    )
    
    
    if not os.path.exists("logs"):   
        os.makedirs("logs")

    train_users = load_fixed_users("../utils/user_train_set.pkl")

    pretrained_belief_path = "pretrained_belief_model.pth"
    if os.path.exists(pretrained_belief_path):
        belief_model.load_state_dict(torch.load(pretrained_belief_path), strict=False)
        print(f"--- Loaded pretrained belief model from {pretrained_belief_path} ---")

    if os.path.exists("value_net_checkpoint.pth"):
        value_net.load_state_dict(torch.load("value_net_checkpoint.pth"))
        belief_model.load_state_dict(torch.load("belief_model_checkpoint.pth"), strict=False)
        if os.path.exists("actor_checkpoint.pth"):
            actor.load_state_dict(torch.load("actor_checkpoint.pth"))
        print("--- Resuming training from existing checkpoint ---")
    
    print("Start")
    start_time = time.time()

    # Train
    train_curio(
        env,
        value_net,
        belief_model,
        actor,
        optimizer,
        belief_optimizer,
        train_users=train_users,
        num_episodes=1000,
    )

    simple_config = { "latent_keys": actor.latent_keys,"input_dim": 8,
        "hidden_dim": 128
    }

    # Create the directory if it doesn't exist
    if not os.path.exists("final_model"):   
        os.makedirs("final_model")
    
    
    with open("final_model/config.json", "w") as f:
        json.dump(simple_config, f, indent=2)
    
    torch.save(value_net.state_dict(), "final_model/value_net_final.pth")
    torch.save(belief_model.state_dict(), "final_model/belief_model_final.pth")
    torch.save(actor.state_dict(), "final_model/actor_final.pth")
    print("Saved final trained models.")

    end_time = time.time()
    total_time = str(timedelta(seconds=int(end_time - start_time)))

    print(f"Total time taken for training: {total_time}")

