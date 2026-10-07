"""Does the auxiliary pairwise (Arena/judge) training set contain cold-start models?"""
from router.config import load_config
from training.data.facade import load_training_data

d = load_training_data(load_config())
cold = set(d.cold_start_model_ids())
print("cold-start models:", len(cold), sorted(cold)[:8], "...")
for scope in ("warm", "all"):
    p = d.pairwise(split="train", models=scope)
    hit = p["model_a"].isin(cold) | p["model_b"].isin(cold)
    print(f"pairwise(split='train', models={scope!r}): {len(p)} battles, "
          f"{int(hit.sum())} involve a cold-start model")
    if scope == "all" and hit.any():
        print("  cold models present:", sorted((set(p.loc[hit, 'model_a']) | set(p.loc[hit, 'model_b'])) & cold))
