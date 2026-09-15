import numpy as np
from typing import Dict, Any, Tuple, List
import torch

from product_scoring import recommend_product
from reward_module import total_reward, dict_to_vec
from generate_users import Simulated_User


class CurioEnvironment:
    """
    Simulation environment for user-agent dialogue and recommendation.
    Tracks state, updates belief based on dialogue, and calculates rewards.
    """

    def __init__(
        self,
        user: Simulated_User,
        user_types: Dict[str, Dict[str, float]],
        product_map: Dict[str, Dict[str, float]],
        belief_model,
        tokenizer,
        max_turns: int = 8,
        alpha: float = 2.0,
        gamma: float = 1.0,
    ):
        self.user = user
        self.user_types = user_types
        self.product_map = product_map
        self.belief_model = belief_model
        self.tokenizer = tokenizer

        self.max_turns = max_turns
        self.alpha = alpha
        self.gamma = gamma

        # episode state
        self.turn = 0
        self.done = False
        self.history: List[Dict[str, str]] = []
        self.asked_keys = set()

        # initial belief state
        self.belief = None
        self.true_type_idx = None


    def _encode_history(self):
        """
        Convert conversation history into model input.
        """
        text = "".join([f"{m['role']}: {m['content']}\n" for m in self.history])

        batch = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        return batch["input_ids"], batch["attention_mask"]

    def _update_belief(self) -> np.ndarray:
      """
      Pass current history through the model return a flat 8D array.
      """
      ids, mask = self._encode_history()
      
      device = next(self.belief_model.parameters()).device

      # get prediction
      with torch.no_grad():
          _, latent, _ = self.belief_model(ids.to(device), mask.to(device))
      
      # extract the first (and only) item in the batch and move to CPU/NumPy
      belief_vec = latent[0].cpu().numpy()
      
      num_types = len(self.user_types)
      assert belief_vec.shape == (num_types,), (
          f"Expected belief over {num_types} user types, got {belief_vec.shape}"
      )
      
      return belief_vec

    def _get_true_type(self) -> int:
        """
        Finds the closest archetype index to the actual simulated user
        """
        user_vec = dict_to_vec(self.user.get_latent_variables())
        type_dist = []
        for _, type_traits in self.user_types.items():
            type_vec = dict_to_vec(type_traits)
            type_dist.append(np.linalg.norm(user_vec - type_vec))
        return int(np.argmin(type_dist))

    def step(self, agent_message: str) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Agent sends a message, user replies, belief updates, reward computed
        """

        if self.done:
            raise RuntimeError("Env is done. Call reset().")

        # update history with agent action
        self.history.append({"role": "assistant", "content": agent_message})
        belief_prev = self.belief

        # get user response
        user_response = self.user.converse_with_user(agent_message, use_llm=False)
        self.history.append({"role": "user", "content": user_response})

        # state change
        belief_next = self._update_belief()
        self.turn += 1

        # end of episode logic
        if self.turn >= self.max_turns:
            self.done = True
            self.belief = belief_next

            # recommend product
            rec_id, _ = recommend_product(
                belief_next,
                self.user_types,
                self.product_map,
            )

            # compute reward
            rewards = total_reward(
                chosen_a=rec_id,
                product_emb=self.product_map,
                true_user_latent=self.user.get_latent_variables(), 
                belief_before=belief_prev if belief_prev is not None else belief_next,
                belief_after=belief_next,
                alpha=self.alpha,
                gamma=self.gamma,
            )

            obs = {
                "history": self.history,
                "belief": belief_next,
                "recommendation": rec_id,
            }

            return obs, rewards["total"], True, rewards


        if belief_prev is None:
            belief_prev = np.full(len(self.user_types), 1.0 / len(self.user_types), dtype=np.float32)
        
        self.belief = belief_next  

        # information gain
        def get_entropy(b):
            return -np.sum(b * np.log(b + 1e-12))
        
        ent_before = get_entropy(belief_prev) if belief_prev is not None else np.log(8)
        ent_after = get_entropy(belief_next)

        prev_acc = float(belief_prev[self.true_type_idx])
        next_acc = float(belief_next[self.true_type_idx])

        
        gain = ent_before - ent_after
        intrinsic = self.alpha * max(0, gain)

        attr = agent_message.split("your ", 1)[-1].rstrip("?").strip().replace(" ", "_")
        if attr in self.asked_keys:
            intrinsic -= 2.0  
        self.asked_keys.add(attr)

        print(f"  True Type Prob: {prev_acc:.4f} -> {next_acc:.4f} (DiffAcc: {gain:.4f})")
        print(f"  Top Belief Prob: {np.max(belief_next):.4f}")
        step_reward = intrinsic

        obs = {
            "history": self.history,
            "belief": belief_next,
        }
        info = {
            "intrinsic": intrinsic,
            "gain": gain
        }

        return obs, step_reward, False, info

    def reset(self, new_user: bool = False):
      if new_user:
          self.user = Simulated_User()

      self.history = []
      self.turn = 0      
      self.done = False  
      self.asked_keys = set()

      n = len(self.user_types)
      self.belief = np.full(n, 1.0 / n, dtype=np.float32)
      self.true_type_idx = self._get_true_type()

      return {"history": [], "belief": self.belief}
