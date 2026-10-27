"""提取 6 个 run 的 TB 曲线到 JSON（供本地绘图与报告）。"""
import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = "/root/acmpc/runs/tb"
RUNS = {
    "horizontal": {"mpc": "acmpc_horizontal_run1_0", "mlp": "acmlp_horizontal_run1_0"},
    "vertical": {"mpc": "acmpc_vertical_run1_0", "mlp": "acmlp_vertical_run1_0"},
    "splits": {"mpc": "acmpc_splits_run1_0", "mlp": "acmlp_splits_run1_0"},
}
TAGS = [
    "eval/gates_passed",
    "eval/success_rate",
    "rollout/ep_rew_mean",
    "train/entropy_loss",
    "train/value_loss",
    "train/std",
    "train/approx_kl",
]

out = {}
for track, pols in RUNS.items():
    out[track] = {}
    for pol, d in pols.items():
        ea = EventAccumulator(os.path.join(TB, d), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        out[track][pol] = {}
        for t in TAGS:
            if t in tags:
                out[track][pol][t] = [[e.step, round(e.value, 6)] for e in ea.Scalars(t)]

with open("/root/acmpc/curves_6runs.json", "w") as f:
    json.dump(out, f)
print("saved /root/acmpc/curves_6runs.json")
for track, pols in out.items():
    print(track, {p: sorted(v.keys()) for p, v in pols.items()})
