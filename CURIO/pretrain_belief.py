import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from curio_belief_model import CurioBeliefModel
from generate_users import Simulated_User
from product_scoring import dict_to_vec, load_user_types


LATENT_KEYS = [
    "comfort",
    "functionality",
    "maintenance",
    "status_symbol",
    "modesty",
    "sustainability",
    "risk_tolerance",
    "trend_sensitivity",
]


def display_att(attr_key):
    return attr_key.replace("_", " ")


def build_history(turns):
    t = ""
    for msg in turns:
        t += f"{msg['role']}: {msg['content']}\n"
    return t


def target(user, user_types, temperature=0.2):
    user_vec = dict_to_vec(user.get_latent_variables())
    distances = []

    for temp, t in user_types.items():
        type_vec = dict_to_vec(t)
        distances.append(np.linalg.norm(user_vec - type_vec))

    distances = torch.tensor(distances, dtype=torch.float32)

    #soft target
    return torch.softmax(-distances / temperature, dim=0)


def generate_dialogue(user, max_turns=8):
    attrs = random.sample(LATENT_KEYS, k=random.randint(1, max_turns))
    history = []

    for attr in attrs:
        question = f"Could you tell me more about your {display_att(attr)}?"

        answer = user.converse_with_user(question, use_llm=False)
        history.append({"role": "assistant", "content": question})
        history.append({"role": "user", "content": answer})

    return build_history(history)


def pretrain_belief(num_examples=8000,
    batch_size=32,
    epochs=3, lr=3e-4,
    seed=42, output_path="pretrained_belief_model.pth",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelName = "sentence-transformers/all-MiniLM-L6-v2"
    user_types = load_user_types("data/user_types.json")

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    model = CurioBeliefModel(num_types=len(user_types), model_name=modelName).to(device)
    model.train()

    optimizer = torch.optim.Adam(list(model.type_head.parameters()) + list(model.latent_head.parameters()), lr=lr)
    type_criterion = nn.KLDivLoss(reduction="batchmean")
    latent_criterion = nn.MSELoss()

    examples = []

    for idx in range(num_examples):
        user = Simulated_User()
        text = generate_dialogue(user)

        type_target = target(user, user_types)
        latent_target = torch.tensor(dict_to_vec(user.get_latent_variables()), dtype=torch.float32)
        
        examples.append((text, type_target, latent_target))


    for epoch in range(epochs):
        losses = []

        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            texts, type_targets, latent_targets = zip(*batch)

            encoded = tokenizer(list(texts), return_tensors="pt",padding=True,truncation=True, max_length=256)
            input_ids = encoded["input_ids"].to(device)

            attention_mask = encoded["attention_mask"].to(device)
            

            logits, temp, latent_pred = model(input_ids, attention_mask)
            #recover soft archetype posterior
            s = torch.log_softmax(logits, dim=-1)
            type_t = torch.stack(type_targets).to(device)
            latent_t = torch.stack(latent_targets).to(device)
            type_loss = type_criterion(s, type_t)
            latent_loss = latent_criterion(latent_pred, latent_t)

            loss = type_loss + 0.05 * latent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        print(f"Epoch {epoch + 1}, loss={np.mean(losses)}")

    torch.save(model.state_dict(), output_path)
    meta_path = os.path.splitext(output_path)[0] + "_meta.json"

    with open(meta_path, "w") as f:
        json.dump(
            { "num_examples": num_examples, "batch_size": batch_size, "epochs": epochs, "lr": lr, "seed": seed,
              "model_name": modelName,},
            f, indent=2,)
    print(f"Finished.")


if __name__ == "__main__":
    pretrain_belief()
