#!/usr/bin/env python3
"""Build the Jev article figures from the frozen experiment artifacts.

The script intentionally depends only on Python's standard library. It emits
SVG so the figures remain inspectable and reproducible; the repository's image
conversion step creates PNG copies for Markdown renderers that do not display
SVG reliably.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "articles" / "figures"
C_SAFE = (
    ROOT
    / "experiments"
    / "jev-calibration"
    / "results"
    / "c-safeqa-matrix-20260927-01"
    / "analysis.json"
)
RLCD = (
    ROOT
    / "experiments"
    / "jev-calibration"
    / "results"
    / "rlcd-distillation-20260927-01"
    / "results.json"
)
BEAVER = (
    ROOT
    / "experiments"
    / "jev-calibration"
    / "results"
    / "beavertails-distilled-20260928-01"
    / "analysis.json"
)

INK = "#102A3A"
MUTED = "#667681"
GRID = "#D7DDD9"
PAPER = "#F6F2E9"
JEV = "#F26B4A"
OPEN = "#5278C7"
TRAINED = "#169C82"
SOFT = "#5278C7"
HARD = "#7B61A8"
HYBRID = "#169C82"


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 24, *,
         anchor: str = "start", fill: str = INK, weight: int = 400,
         family: str = "Inter, Arial, sans-serif") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, color: str = GRID,
         width: float = 2, dash: str | None = None, opacity: float = 1) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dashed}/>'
    )


def rect(x: float, y: float, width: float, height: float, *, fill: str,
         radius: float = 0, opacity: float = 1, stroke: str | None = None) -> str:
    outline = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" opacity="{opacity}"{outline}/>'
    )


def circle(x: float, y: float, radius: float, *, fill: str,
           stroke: str = PAPER, width: float = 4) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def svg_document(width: int, height: int, body: list[str], title: str,
                 description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{esc(title)}</title>",
            f"<desc id=\"desc\">{esc(description)}</desc>",
            rect(0, 0, width, height, fill=PAPER),
            *body,
            "</svg>",
        ]
    )


def write_svg(name: str, width: int, height: int, body: list[str], title: str,
              description: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(
        svg_document(width, height, body, title, description), encoding="utf-8"
    )


def make_study_arc(csafe: dict, rlcd: dict, beaver: dict) -> None:
    r = {
        (row["model_key"], row["system"]): row["official_on_teacher_paired"]
        for row in rlcd["final_systems"]
    }
    b = beaver["systems"]
    c = csafe["common_metrics"]["pooled"]
    groups = [
        (
            "RLCDAlignBench",
            "1,174 held-out rows · pooled",
            rlcd["jev_reference_on_teacher_paired"]["auroc"],
            r[("qwen35-9b", "base")]["auroc"],
            r[("qwen35-4b", "hybrid")]["auroc"],
            "Qwen 9B",
            "Qwen 4B + hybrid",
        ),
        (
            "BeaverTails",
            "2,000 external rows · pooled",
            b["jev_live"]["auroc"],
            b["qwen35_9b_base"]["auroc"],
            b["qwen35_4b_jev"]["auroc"],
            "Qwen 9B",
            "Qwen 4B + Jev",
        ),
        (
            "C-SafeQA",
            "1,931 common rows · pooled",
            c["jev_live"]["auroc"],
            c["qwen35_9b_base"]["auroc"],
            c["qwen35_4b_hard"]["auroc"],
            "Qwen 9B",
            "Qwen 4B + hard",
        ),
    ]

    width, height = 1400, 900
    left, right, top, bottom = 125, 55, 150, 250
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.5, 1.0

    def y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    body: list[str] = [
        text(70, 68, "Training erased Jev's point-estimate lead", 42, weight=700,
             family="Georgia, serif"),
        text(70, 108, "AUROC across frozen held-out and external tests", 24, fill=MUTED),
    ]
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        yy = y(tick)
        body.extend([
            line(left, yy, width - right, yy, color=GRID, width=1),
            text(left - 18, yy + 8, f"{tick:.1f}", 20, anchor="end", fill=MUTED),
        ])
    body.append(text(38, top + plot_h / 2, "AUROC", 20, fill=MUTED))

    centers = [330, 700, 1070]
    bar_w, gap = 70, 18
    baseline_y = y(ymin)
    for center, (dataset, subtitle, jev, untouched, trained, untouched_name, trained_name) in zip(centers, groups):
        values = [(jev, JEV), (untouched, OPEN), (trained, TRAINED)]
        xs = [center - bar_w - gap, center, center + bar_w + gap]
        for xx, (value, color) in zip(xs, values):
            if value is None:
                body.extend([
                    line(xx - 22, baseline_y - 20, xx + 22, baseline_y - 20,
                         color=MUTED, width=3),
                    text(xx, baseline_y - 42, "not run", 16, anchor="middle", fill=MUTED),
                ])
                continue
            yy = y(value)
            body.extend([
                rect(xx - bar_w / 2, yy, bar_w, baseline_y - yy,
                     fill=color, radius=10),
                text(xx, yy - 14, f"{value:.3f}", 21, anchor="middle", weight=700),
            ])
        body.extend([
            text(center, baseline_y + 47, dataset, 25, anchor="middle", weight=700),
            text(center, baseline_y + 78, subtitle, 17, anchor="middle", fill=MUTED),
        ])
        model_note = f"untouched: {untouched_name} · trained: {trained_name}"
        body.append(text(center, baseline_y + 110, model_note, 16,
                         anchor="middle", fill=MUTED, weight=600))

    legend_y = 825
    for x0, color, label in [
        (185, JEV, "Jev"),
        (390, OPEN, "best untouched open model"),
        (765, TRAINED, "best trained 4B student"),
    ]:
        body.extend([
            rect(x0, legend_y - 16, 26, 26, fill=color, radius=6),
            text(x0 + 40, legend_y + 6, label, 20, fill=INK, weight=600),
        ])
    body.append(text(
        width - 60, 890,
        "All bars use the exact common rows within each dataset; datasets are not directly comparable.",
        15, anchor="end", fill=MUTED,
    ))
    write_svg(
        "jev-study-arc.svg", width, height, body,
        "Training erased Jev's point-estimate lead",
        "Grouped AUROC bars show Jev ahead of the best untouched model on each dataset, "
        "while a frozen trained 4B student has a higher point estimate on every held-out or external test.",
    )


def make_transfer_intervals(csafe: dict) -> None:
    bootstrap = csafe["paired_prompt_cluster_bootstrap"]
    rows = [
        ("Qwen 4B + Jev", "jev_live_vs_qwen35_4b_jev", SOFT),
        ("Qwen 4B + hard", "jev_live_vs_qwen35_4b_hard", HARD),
        ("Qwen 4B + hybrid", "jev_live_vs_qwen35_4b_hybrid", HYBRID),
        ("Qwen 9B + Jev", "jev_live_vs_qwen35_9b_jev", SOFT),
        ("Qwen 9B + hard", "jev_live_vs_qwen35_9b_hard", HARD),
        ("Qwen 9B + hybrid", "jev_live_vs_qwen35_9b_hybrid", HYBRID),
    ]
    metrics = [
        ("AUROC gain", "auroc_left_minus_right", 0.14),
        ("Log-loss reduction", "log_loss_gain_right_minus_left", 0.13),
        ("Brier reduction", "brier_gain_right_minus_left", 0.035),
    ]
    width, height = 1500, 920
    left_label, panel_w, panel_gap = 255, 350, 60
    top, row_gap = 245, 82
    body: list[str] = [
        text(70, 68, "Every student beat Jev on ranking and proper scores", 42,
             weight=700, family="Georgia, serif"),
        text(70, 108, "C-SafeQA · student improvement over live Jev · 95% prompt-cluster bootstrap intervals",
             23, fill=MUTED),
    ]
    for idx, (label, _, color) in enumerate(rows):
        yy = top + idx * row_gap
        body.append(text(left_label - 24, yy + 7, label, 20, anchor="end", weight=600))
        body.append(circle(left_label - 8, yy, 7, fill=color, stroke=color, width=0))
    for panel_idx, (title, key, xmax) in enumerate(metrics):
        x0 = left_label + 28 + panel_idx * (panel_w + panel_gap)
        x1 = x0 + panel_w
        body.extend([
            text((x0 + x1) / 2, 175, title, 23, anchor="middle", weight=700),
            line(x0, top - 40, x0, top + (len(rows) - 1) * row_gap + 34,
                 color=INK, width=2),
        ])
        for tick_idx in range(4):
            value = xmax * tick_idx / 3
            xx = x0 + panel_w * tick_idx / 3
            body.extend([
                line(xx, top - 32, xx, top + (len(rows) - 1) * row_gap + 30,
                     color=GRID, width=1),
                text(xx, top + len(rows) * row_gap + 4, f"{value:.3f}", 17,
                     anchor="middle", fill=MUTED),
            ])
        for row_idx, (_, bootstrap_key, color) in enumerate(rows):
            entry = bootstrap[bootstrap_key]
            point_raw = entry["point"][key]
            low_raw, high_raw = entry["interval_95"][key]
            # Stored comparisons are Jev (left) minus student (right), so all
            # three registered improvement measures become intuitive after a
            # sign flip.
            point_value = -point_raw
            low, high = -high_raw, -low_raw
            yy = top + row_idx * row_gap
            xp = x0 + panel_w * point_value / xmax
            xl = x0 + panel_w * low / xmax
            xh = x0 + panel_w * high / xmax
            body.extend([
                line(xl, yy, xh, yy, color=color, width=7),
                line(xl, yy - 10, xl, yy + 10, color=color, width=3),
                line(xh, yy - 10, xh, yy + 10, color=color, width=3),
                circle(xp, yy, 9, fill=color, stroke=PAPER, width=3),
            ])
    legend_y = 850
    for x0, color, label in [
        (430, SOFT, "Jev soft targets"),
        (725, HARD, "official hard labels"),
        (1035, HYBRID, "50/50 hybrid"),
    ]:
        body.extend([circle(x0, legend_y, 8, fill=color, stroke=color, width=0),
                     text(x0 + 20, legend_y + 7, label, 18, weight=600)])
    body.append(text(1440, 900, "Right is better; zero means no improvement over Jev.",
                     16, anchor="end", fill=MUTED))
    write_svg(
        "jev-csafeqa-transfer-intervals.svg", width, height, body,
        "Every student beat Jev on ranking and proper scores",
        "Six Qwen students show positive AUROC, log-loss and Brier improvements over Jev, "
        "with every 95 percent cluster-bootstrap interval excluding zero.",
    )


def make_operating_points(csafe: dict) -> None:
    systems = csafe["common_metrics"]["pooled"]
    points = [
        ("Jev", "jev_live", JEV, 14, -18, -18),
        ("Qwen 4B base", "qwen35_4b_base", OPEN, 11, -18, 34),
        ("4B + Jev", "qwen35_4b_jev", SOFT, 12, 18, -18),
        ("4B + hybrid", "qwen35_4b_hybrid", HYBRID, 12, 20, 30),
        ("4B + hard", "qwen35_4b_hard", HARD, 12, -20, -22),
        ("9B + hard", "qwen35_9b_hard", "#B58735", 12, 18, -18),
    ]
    width, height = 1300, 900
    left, right, top, bottom = 160, 90, 150, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 0.85, 1.0, 0.25, 0.9

    def xy(system_key: str) -> tuple[float, float]:
        rates = systems[system_key]["threshold_rates"]
        sx = rates["safe_specificity"]
        ry = rates["unsafe_recall"]
        return (
            left + (sx - xmin) / (xmax - xmin) * plot_w,
            top + (ymax - ry) / (ymax - ymin) * plot_h,
        )

    body: list[str] = [
        text(70, 68, "Accuracy hid Jev’s conservative operating point", 42,
             weight=700, family="Georgia, serif"),
        text(70, 108, "C-SafeQA at a fixed 0.5 threshold · higher and farther right are better",
             23, fill=MUTED),
    ]
    for tick in [0.85, 0.90, 0.95, 1.0]:
        xx = left + (tick - xmin) / (xmax - xmin) * plot_w
        body.extend([
            line(xx, top, xx, top + plot_h, color=GRID, width=1),
            text(xx, top + plot_h + 42, f"{tick:.0%}", 20, anchor="middle", fill=MUTED),
        ])
    for tick in [0.25, 0.40, 0.55, 0.70, 0.85]:
        yy = top + (ymax - tick) / (ymax - ymin) * plot_h
        body.extend([
            line(left, yy, left + plot_w, yy, color=GRID, width=1),
            text(left - 20, yy + 7, f"{tick:.0%}", 20, anchor="end", fill=MUTED),
        ])
    body.extend([
        text(left + plot_w / 2, height - 58, "Safe specificity — correctly accepted safe responses",
             22, anchor="middle", weight=600),
        text(36, top + plot_h / 2, "Unsafe recall", 22, fill=MUTED, weight=600),
        text(left + plot_w - 12, top + 34, "ideal", 17, anchor="end", fill=MUTED, weight=700),
    ])
    base_x, base_y = xy("qwen35_4b_base")
    for target in ["qwen35_4b_jev", "qwen35_4b_hybrid", "qwen35_4b_hard"]:
        tx, ty = xy(target)
        body.append(line(base_x, base_y, tx, ty, color=OPEN, width=2,
                         dash="8 8", opacity=0.55))
    for label, key, color, radius, dx, dy in points:
        xx, yy = xy(key)
        anchor = "start" if dx >= 0 else "end"
        body.extend([
            circle(xx, yy, radius, fill=color, stroke=PAPER, width=5),
            text(xx + dx, yy + dy, label, 20, anchor=anchor, weight=700, fill=INK),
        ])
    body.extend([
        rect(250, 662, 430, 78, fill="#EAE4D8", radius=14),
        text(272, 692, "Jev: 33.6% unsafe recall", 19, weight=700, fill=JEV),
        text(272, 721, "at 98.8% safe specificity", 18, fill=MUTED),
    ])
    write_svg(
        "jev-csafeqa-operating-points.svg", width, height, body,
        "Accuracy hid Jev's conservative operating point",
        "Scatter plot of unsafe recall against safe specificity. Jev has very high specificity "
        "but low recall, while trained 4B students detect many more unsafe responses.",
    )


def main() -> None:
    csafe = load(C_SAFE)
    rlcd = load(RLCD)
    beaver = load(BEAVER)
    make_study_arc(csafe, rlcd, beaver)
    make_transfer_intervals(csafe)
    make_operating_points(csafe)
    for path in sorted(FIGURES.glob("jev-*.svg")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
