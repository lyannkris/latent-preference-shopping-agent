# VPL-only Sequential Active Baseline

This folder contains the final VPL-only baseline used for comparison with `our_model`.

## Final baseline

The selected model is a sequential active VPL-only baseline with `K=10` pairwise preference observations.

At test time, the model does not use CURIO questions, natural-language dialogue, PPO, or an `our_model` question policy. It sequentially selects product pairs by greedy expected posterior entropy minimization, receives an A/B preference label from the user simulator, updates the posterior over the 8D latent variable, and ranks products with the learned reward model.

## Folder structure

- `data.py`: Builds VPL-style pairwise preference episodes from users and product embeddings.
- `model.py`: Variational posterior and latent-conditioned reward model.
- `train.py`: Trains the VPL-only model with pairwise preference loss and KL regularization.
- `eval.py`: Evaluates fixed random-context pairwise observations.
- `active_eval.py`: Evaluates sequential active pairwise query selection.
- `results/`: Final trained model, run config, training history, datasets, and metrics.
- `utils/`: All data dependencies used by this baseline.

## Data files

- `utils/user_train_set.pkl`: Team-provided training users.
- `utils/user_eval_set.pkl`: Team-provided evaluation users.
- `utils/user_test_set.pkl`: Team-provided test users.
- `utils/product_embeddings.json`: Product 8D embeddings.
- `utils/generate_users.py`: Class/function dependency needed to unpickle and interpret the simulated users.
- `utils/shared_user_generator.py`: Loader that converts pickle users into the normalized user format used by VPL-only.

## Final result file

Use this file as the main reported result:

```bash
results/active_test_results.json
```

The fixed random-context result is kept only as a sanity check:

```bash
results/test_results.json
```

## Re-run evaluation

From this folder:

```bash
python3 eval.py \
  --results-dir /Users/ee/Documents/UMich/Course/2026WN/EECS545/Project/VPL/VPL/results \
  --scoring-mode reward_model
```

```bash
python3 active_eval.py \
  --base-results-dir /Users/ee/Documents/UMich/Course/2026WN/EECS545/Project/VPL/VPL/results \
  --output-path /Users/ee/Documents/UMich/Course/2026WN/EECS545/Project/VPL/VPL/results/active_test_results.json \
  --num-context-pairs 10 \
  --max-candidate-pairs 2000 \
  --scoring-mode reward_model
```

## Final metrics

Sequential active VPL-only, `K=10`:

| Metric | Value |
| --- | ---: |
| Avg utility | 0.9724 |
| Avg regret | 0.0186 |
| Avg normalized regret | 0.0588 |
| Threshold success | 0.9120 |
| Success@1 | 0.0240 |
| Success@3 | 0.0560 |
| Success@5 | 0.1040 |
| Success@10 | 0.1440 |
| Success@50 | 0.2400 |
| Success@100 | 0.2880 |
| Success@200 | 0.3360 |
| Success@500 | 0.5600 |
| Success@1000 | 0.7360 |
| Avg context pairs | 10.0000 |
| Avg posterior entropy after | 3.5955 |
| Avg uncertainty reduction | 0.7756 |
