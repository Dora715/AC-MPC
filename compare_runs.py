"""对比 6 个 run 的 eval 曲线：过门数 / 完赛率 / 训练奖励，按 250k 窗口汇总。"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_ROOT = "/root/acmpc/runs/tb"
RUNS = ["acmpc_horizontal_run1_0", "acmlp_horizontal_run1_0",
        "acmpc_vertical_run1_0", "acmlp_vertical_run1_0",
        "acmpc_splits_run1_0", "acmlp_splits_run1_0"]

WIN = 250_000  # 汇总窗口（步）


def load(dirpath):
    ea = EventAccumulator(dirpath, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for tag in ("eval/gates_passed", "eval/success_rate", "rollout/ep_rew_mean"):
        if tag in tags:
            out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def summarize(events, n_win):
    """把 (step, val) 序列按窗口平均，输出 [(窗口末步, 均值)]。"""
    res = []
    for i in range(0, len(events), n_win):
        chunk = events[i:i + n_win]
        res.append((chunk[-1][0], sum(v for _, v in chunk) / len(chunk)))
    return res


def main():
    first_success = {}
    for name in RUNS:
        data = load(os.path.join(TB_ROOT, name))
        label = name.replace("_run1_0", "")
        print(f"\n===== {label} =====")
        for tag, events in data.items():
            print(f"  {tag}: {len(events)} pts, step {events[0][0]}..{events[-1][0]}")
        gates = data.get("eval/gates_passed", [])
        succ = data.get("eval/success_rate", [])
        rew = data.get("rollout/ep_rew_mean", [])
        n_win = max(1, len(gates) // 8) if gates else 1
        g = summarize(gates, n_win)
        s = summarize(succ, n_win)
        r = summarize(rew, max(1, len(rew) // 8))
        print(f"  {'step':>9} {'gates':>6} {'succ':>6} {'ep_rew':>8}")
        for i in range(max(len(g), len(s))):
            row = f"  {g[i][0]:>9} "
            row += f"{g[i][1]:>6.2f} " if i < len(g) else "   n/a "
            row += f"{s[i][1]:>6.2f} " if i < len(s) else "   n/a "
            row += f"{r[i][1]:>8.2f}" if i < len(r) else "     n/a"
            print(row)
        fs = next((st for st, v in succ if v > 0), None)
        first_success[label] = fs
        print(f"  first success at step: {fs}")
    print("\n===== FIRST SUCCESS SUMMARY =====")
    for k, v in sorted(first_success.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
