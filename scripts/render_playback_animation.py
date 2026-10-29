"""Animate exported patient power blocks, QC and frozen-source assessment scores."""

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]


def generate_playback_gif(subject_id="Nekrasova_P36", output_path=ROOT / "docs/bedside_playback_demo.gif",
                          n_frames=40, stride=None):
    path = ROOT / f"results/bedside_playback_demo/playback_trajectory_{subject_id}.json"
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    steps = trajectory["steps"]
    if not steps or n_frames < 1:
        raise ValueError("Animation needs a nonempty trace and positive frame count.")
    frames = np.unique(np.linspace(0, len(steps)-1, min(n_frames, len(steps)), dtype=int))
    points = [s for s in steps if s["score"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="#0b111e")
    for ax in axes:
        ax.set_facecolor("#141c2e")
        ax.tick_params(colors="#a4b1c5", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#354458")
        ax.grid(axis="y", color="#27354b", linewidth=0.5)
    power_ax, score_ax, feature_ax = axes
    bars = power_ax.bar(np.arange(6), np.zeros(6), color="#57b8ff")
    power_ax.set_xticks(range(6), ["F3", "F4", "C3", "C4", "O1", "O2"])
    max_power = max(np.log10(1+np.asarray(s["theta_power"])).max() for s in steps)
    power_ax.set_ylim(0, max(1, max_power)*1.1)
    power_ax.set_ylabel("log10(1 + cached theta power)", color="#a4b1c5", fontsize=9)
    power_ax.set_title("Actual cached power block", color="#f0f4f8", fontsize=10)
    raw_line, = score_ax.plot([], [], color="#57b8ff", label="S(x)")
    clip_line, = score_ax.plot([], [], color="#69ddb4", label="S1.5(x)")
    score_ax.axhline(trajectory["threshold_source_train"], color="#57b8ff", ls="--", lw=1, label="Source median S")
    score_ax.axhline(trajectory["threshold_source_train_c15"], color="#69ddb4", ls="--", lw=1, label="Source median S1.5")
    values = [s[k] for s in points for k in ("score", "score_c15")]
    values += [trajectory["threshold_source_train"], trajectory["threshold_source_train_c15"]]
    score_ax.set_ylim(min(-.3, min(values))-.2, max(.3, max(values))+.2)
    score_ax.set_xlim(0, len(steps))
    score_ax.set_title("Recorded assessment scores", color="#f0f4f8", fontsize=10)
    rho_line, = feature_ax.plot([], [], color="#57b8ff", label="Lag-1 persistence")
    res_line, = feature_ax.plot([], [], color="#d29dff", label="Residual correlation")
    feature_ax.set_xlim(0, len(steps))
    feature_ax.set_ylim(-1.05, 1.05)
    feature_ax.set_title("Cumulative features", color="#f0f4f8", fontsize=10)
    for ax in (score_ax, feature_ax):
        ax.set_xlabel("Recording block (4 s)", color="#a4b1c5", fontsize=9)
        ax.legend(loc="lower right", fontsize=7, facecolor="#141c2e", edgecolor="#354458", labelcolor="#f0f4f8")
    fig.suptitle(f"Cached-power replay | {subject_id} | frozen source model", color="#f0f4f8", fontsize=12)
    status = fig.text(.5, .07, "", ha="center", color="#ffcf72", fontsize=9)
    fig.text(.5, .015, "Source-training engineering example; power display, no raw EEG waveforms. Dots/lines use actual evaluation events.", ha="center", color="#a4b1c5", fontsize=8)
    fig.subplots_adjust(left=.055, right=.985, bottom=.22, top=.83, wspace=.3)

    def update(frame_idx):
        step = steps[int(frame_idx)]
        for bar, value in zip(bars, np.log10(1+np.asarray(step["theta_power"]))):
            bar.set_height(value)
            bar.set_color("#57b8ff" if step["is_clean"] else "#ffcf72")
        visible = [s for s in points if s["epoch_idx"] <= step["epoch_idx"]]
        x = [s["epoch_idx"]+1 for s in visible]
        raw_line.set_data(x, [s["score"] for s in visible])
        clip_line.set_data(x, [s["score_c15"] for s in visible])
        rho_line.set_data(x, [s["features"]["lag1"] for s in visible])
        res_line.set_data(x, [s["features"]["residual_corr"] for s in visible])
        state = step["prioritization_state"]
        if step["score"] is not None:
            state = "PRIORITIZE_FOR_MULTIMODAL" if step["score"] >= trajectory["threshold_source_train"] else "LOWER_PRIORITY"
        status.set_text(f"Block {step['epoch_idx']+1}/{len(steps)} | clean {step['total_clean_blocks']} | rejected {step['rejected_blocks']} | {state}")
        return [*bars, raw_line, clip_line, rho_line, res_line, status]

    animation = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer="pillow", fps=8, dpi=100)
    update(frames[-1])
    fig.savefig(output_path.with_suffix(".png"), dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Rendered {len(frames)} recorded-power frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="Nekrasova_P36")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/bedside_playback_demo.gif")
    args = parser.parse_args()
    generate_playback_gif(args.subject, args.output, args.frames)
