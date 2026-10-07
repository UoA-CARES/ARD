#!/usr/bin/env python3
"""
Draw the genealogy of every seed run: which candidate came from which parent.

Every run of `main.py` writes one `reward_history.json` under
`<runs-dir>/seed_<n>/<task>/`. A `run` record is one reward candidate and carries
`parent_tag` -- the survivor whose conversation branch it was generated from.
Those links form a forest: iteration 1 has no parents (10 roots),
and every later candidate hangs off one earlier candidate. A parent can keep
producing children for many iterations, because survivors stay in the pool and
compete against their own children (see `select_top_k` in src/evaluation/scorer.py),
so an edge may span several iterations.

A candidate is not one training any more. It is trained `repeats` times, each on
its own RL seed, and those trainings are stored as separate `trial` records that
are *not* nodes of the genealogy: they have no `parent_tag` and no children, and
this script skips them. What ranks the candidate is the IQM of its trials -- the
interquartile mean, i.e. throw away the highest and the lowest score and average
what is left, so at the default `repeats: 3` it is the middle training (see
`aggregate_trials` in src/evaluation/scorer.py). Every one of those scores is
kept in `trial_fitnesses`, and both figures show them next to the aggregate, so
a candidate that only looks good because one seed got lucky is visible as such.

Every seed is drawn on one shared fitness scale (same y range, same colour
ramp, same marker sizes), built from all the seeds that were loaded. A seed
whose best candidate is weak therefore peaks lower on the page and stays paler
than a strong one, instead of each figure being rescaled to its own best.

Two figures are written:

  lineage_fitness.png       One panel per seed. x = iteration, y = fitness (log),
                            one dot per candidate at its IQM, a pale bar through
                            it spanning that candidate's repeats (one tick per
                            training), and a line from each parent to each of
                            its children. Blue dots produced at least one child
                            (they were survivors); grey dots are dead ends; the
                            orange path is the ancestry of the run's best
                            candidate. Candidates whose every repeat failed
                            (`fitness: null`) sit on the dashed floor line as
                            grey crosses.

  lineage_tree_seed_<n>.png One file per seed. The same forest as a plain family
                            tree: one row per candidate, indented by iteration,
                            elbow connectors from parent to child. Node colour and
                            size are the fitness (light/small = low, dark/big =
                            high), on the shared scale, and the bracket after
                            each label lists that candidate's repeats. This is
                            the figure to read when the question is purely "who
                            is whose child".

Usage:
    python scripts/plot_lineage.py
    python scripts/plot_lineage.py --seeds 42 47 --out-dir /tmp/figs
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK = "Isaac-ARD-Repose-Cube-Shadow-Direct-v0"
DEFAULT_SEEDS = [42,43,44,45,46]

# Palette tokens shared with scripts/plot_seed_comparison.py.
BLUE = "#2a78d6"        # categorical slot 1 -- candidates that became parents
ORANGE = "#eb6834"      # categorical slot 2 -- the best candidate and its ancestry
MUTED = "#8f8d85"       # dead-end candidates and failed jobs
EDGE = "#c2c0b8"        # parent -> child links
GRID_COLOR = "#dedcd7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
# Fitness ramp for the tree nodes. A single-hue ramp puts almost every candidate
# in the same mid tone, because the values are bimodal: most sit between 0.03 and
# 2 and a handful reach 30. A perceptual heat map spreads that crowded low range
# over several hues while staying monotonic in lightness, so it still reads as an
# ordered scale (and still works in greyscale or for colour-blind readers). The
# ends are trimmed off: raw magma_r starts near white (invisible on the surface)
# and ends at pure black (reads as text).
DEFAULT_CMAP = "magma_r"
CMAP_TRIM = (0.05, 0.92)
# Marker area for the lowest and highest fitness. Fitness is encoded twice, by
# colour and by size, so the winners are findable at a glance and the figure
# survives being printed in greyscale.
SIZE_RANGE = (28, 210)
# Rough width of one character of a tree row label, and of the tree axes, both
# in points. Only used to reserve enough right margin for the longest label.
LABEL_CHAR_PT = 4.4     # at fontsize 7.5
AXES_PT = 540.0         # ~7.5in of the 9.5in figure, once the colorbar is out


def fitness_cmap(name=DEFAULT_CMAP, trim=CMAP_TRIM):
    """`name` with its extreme ends trimmed, so no step vanishes into the page."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trim", base(np.linspace(trim[0], trim[1], 256))
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plot_lineage")


# ------------------------------------------------------------------- loading
def load_history(runs_dir, seed, task):
    """
    Return the record list of one seed run, or None if it has no history.

    ``main.py`` writes the history under ``<task>/<timestamp>/reward_history.json``
    (the timestamp is a per-execution dir, see ``run_refinement``), and older
    layouts dropped that level -- or the task level entirely -- so a fixed path
    is too brittle. Search the whole seed directory for any history and take the
    most recently written one: silently skipping a seed would drop it from a
    figure whose whole point is comparing every seed on one scale.
    """
    seed_dir = Path(runs_dir) / f"seed_{seed}"
    if not seed_dir.is_dir():
        logger.warning(f"No history for seed {seed}: {seed_dir}")
        return None
    paths = sorted(seed_dir.glob("**/reward_history.json"),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        logger.warning(f"No history for seed {seed}: {seed_dir}")
        return None
    path = paths[-1]
    with open(path, "r") as f:
        records = json.load(f)
    logger.info(f"seed {seed}: {len(records)} records from {path}")
    infer_parents(records)
    return records


def short(tag):
    """`iter3_run_7` -> `3.7`, the label used on the tree rows."""
    it, _, run = tag.partition("_run_")
    return f"{it.replace('iter', '')}.{run}"


def infer_parents(records):
    """
    Fill in `parent_tag` on histories written before that field existed.

    Those runs used the greedy hand-off this repo replaced in `5f6d5b3`: one
    conversation was carried forward, and only the winner of iteration i was fed
    back into it, so every candidate of iteration i+1 was generated from that one
    winner. The genealogy is therefore not lost, it is implied: parent = the
    highest-fitness candidate of the previous iteration, ties going to the lowest
    index, which is what `select_best` picked (`max` keeps the first maximum, and
    the batch was ordered by index).

    Each filled record is marked `parent_inferred`, so the figures can say the
    links were reconstructed rather than recorded. Does nothing to a history that
    already has real parents.
    """
    runs = [r for r in records if r["phase"] == "run"]
    if not runs or any(r.get("parent_tag") for r in runs):
        return
    if len({r["iteration"] for r in runs}) < 2:
        return          # one iteration: every candidate is a root either way,
                        # so there is nothing to reconstruct and nothing to flag
    winners = {}
    for r in sorted(runs, key=lambda r: r["index"]):
        it, fit = r["iteration"], r["fitness"]
        if fit is not None and (it not in winners or fit > winners[it]["fitness"]):
            winners[it] = r
    for r in runs:
        parent = winners.get(r["iteration"] - 1)
        r["parent_tag"] = parent["tag"] if parent else None
        r["parent_inferred"] = True
    logger.info(f"  no parent_tag in this history: lineage reconstructed from the "
                f"per-iteration winners ({len(winners)} of them)")


def is_inferred(by_tag):
    """True if this seed's parent links were reconstructed, not recorded."""
    return any(r.get("parent_inferred") for r in by_tag.values())


def trial_values(rec):
    """
    One candidate's repeats, ascending, or `[]` if it has none to show.

    Empty in three cases that all mean the same thing here, "there is no spread
    to show": a history written before `repeats` existed (no key at all), a
    candidate whose every training failed (empty list), and one score, where the
    aggregate *is* that training and repeating it would only clutter the figure.
    """
    vals = rec.get("trial_fitnesses") or []
    return vals if len(vals) > 1 else []


def build_forest(records):
    """
    Index the run-phase records into a parent/child forest.

    Two other phases are skipped. Eval-phase records are re-trainings of the
    winner on extra seeds, not new candidates. Trial-phase records are the
    `repeats` trainings of one candidate; they belong to the run record named in
    their `candidate_tag`, which already carries their scores in
    `trial_fitnesses`, so they are not nodes of the tree either. Neither phase
    has a `parent_tag`, so including them would only add roots.
    """
    runs = [r for r in records if r["phase"] == "run"]
    by_tag = {r["tag"]: r for r in runs}
    children = {t: [] for t in by_tag}
    roots = []
    for r in sorted(runs, key=lambda r: (r["iteration"], r["index"])):
        parent = r["parent_tag"]
        if parent in children:
            children[parent].append(r["tag"])
        else:
            roots.append(r["tag"])
    return by_tag, children, roots


def best_record(by_tag):
    """
    The candidate `select_best` marked, or the top-fitness one if the run is
    still going (the flag is only written after the loop finishes).
    """
    flagged = [r for r in by_tag.values() if r.get("selected_best")]
    if flagged:
        return flagged[0]
    scored = [r for r in by_tag.values() if r["fitness"] is not None]
    return max(scored, key=lambda r: r["fitness"]) if scored else None


def ancestry(by_tag, tag):
    """Tags from the root down to `tag`, following `parent_tag` upwards."""
    chain = []
    while tag in by_tag:
        chain.append(tag)
        tag = by_tag[tag]["parent_tag"]
    return list(reversed(chain))


# -------------------------------------------------------------------- drawing
def style_axes(ax, title, xlabel, ylabel, logy=False):
    """Recessive grid and axes, so the data marks carry the chart."""
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    if logy:
        ax.set_yscale("log")


def node_xy(rec, n_samples, floor):
    """
    Position of one candidate in the fitness panel.

    Candidates of the same iteration share an x, so they are fanned out by index
    to keep the parent links apart. A failed job has no fitness and is parked on
    the floor line rather than dropped, so its edge from the parent still shows.
    """
    x = rec["iteration"] + (rec["index"] - (n_samples - 1) / 2) * 0.055
    y = rec["fitness"] if rec["fitness"] is not None else floor
    return x, y


def plot_fitness_lineage(ax, seed, records, floor=None, ylim=None):
    """
    One seed's forest drawn over (iteration, fitness).

    `floor` and `ylim` are computed once over every seed, so a panel is not
    rescaled to its own best: a weaker seed's peak sits lower on the page than
    a stronger one's, which is the whole point of putting them side by side.
    """
    by_tag, children, _ = build_forest(records)
    scored = [r["fitness"] for r in by_tag.values() if r["fitness"] is not None]
    n_samples = max(r["index"] for r in by_tag.values()) + 1
    if floor is None:
        floor = min(scored) * 0.45 if scored else 0.01
    best = best_record(by_tag)
    line = set(ancestry(by_tag, best["tag"])) if best else set()

    pos = {t: node_xy(r, n_samples, floor) for t, r in by_tag.items()}

    # The repeats behind each dot: a pale bar from the candidate's worst to its
    # best training, one tick per training. The dot is their IQM, so it sits
    # inside the bar and usually below the top of it. A tall bar means the
    # candidate's score depended heavily on the RL seed, which is exactly what
    # the trimming is there to stop from deciding the ranking.
    for tag, r in by_tag.items():
        vals = trial_values(r)
        if len(vals) < 2:
            continue
        x, _ = pos[tag]
        ax.plot([x, x], [min(vals), max(vals)], color=EDGE, linewidth=3.4,
                alpha=0.75, solid_capstyle="round", zorder=1)
        ax.scatter([x] * len(vals), vals, s=26, marker="_", c=MUTED,
                   linewidths=0.9, zorder=2)

    # Edges next, so every dot sits on top of its own links.
    for tag, kids in children.items():
        for kid in kids:
            on_line = tag in line and kid in line
            ax.plot(*zip(pos[tag], pos[kid]),
                    color=ORANGE if on_line else EDGE,
                    linewidth=1.8 if on_line else 0.7,
                    alpha=0.9 if on_line else 0.55,
                    zorder=3 if on_line else 1)

    groups = {"parent": [], "leaf": [], "failed": []}
    for tag, r in by_tag.items():
        key = "failed" if r["fitness"] is None else ("parent" if children[tag] else "leaf")
        groups[key].append(pos[tag])
    for key, color, size, marker, z in (
        ("leaf", MUTED, 22, "o", 2),
        ("parent", BLUE, 40, "o", 4),
        ("failed", MUTED, 34, "x", 2),
    ):
        if groups[key]:
            xs, ys = zip(*groups[key])
            ax.scatter(xs, ys, s=size, c=color, marker=marker, zorder=z,
                       linewidths=1.4 if marker == "x" else 1.2,
                       edgecolors=SURFACE if marker == "o" else color)

    if best:
        bx, by = pos[best["tag"]]
        ax.scatter([bx], [by], s=170, marker="*", color=ORANGE, zorder=5,
                   edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(f"{short(best['tag'])}  {best['fitness']:.2f}", (bx, by),
                    textcoords="offset points", xytext=(9, 6),
                    fontsize=9, color=TEXT_PRIMARY, fontweight="bold", zorder=6)

    if scored:
        ax.axhline(floor, color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
        ax.annotate("failed", (min(r["iteration"] for r in by_tag.values()) - 0.45, floor),
                    textcoords="offset points", xytext=(0, 4),
                    fontsize=8, color=TEXT_SECONDARY)

    iters = sorted({r["iteration"] for r in by_tag.values()})
    n_par = len(groups["parent"])
    note = ", lineage reconstructed" if is_inferred(by_tag) else ""
    # A pre-`repeats` history has one training per candidate and no IQM, so the
    # axis must not claim one.
    has_reps = any(trial_values(r) for r in by_tag.values())
    style_axes(ax, f"seed {seed}  ({len(by_tag)} candidates, "
                   f"{n_par} became parents{note})",
               "iteration",
               "fitness: IQM of the repeats (log)" if has_reps else "fitness (log)",
               logy=True)
    ax.set_xticks(iters)
    ax.set_xlim(iters[0] - 0.5, iters[-1] + 0.6)
    if ylim:
        ax.set_ylim(*ylim)


def layout_tree(by_tag, children, roots):
    """
    One row per candidate, pre-order depth-first: a parent sits directly above
    its own subtree, which is what makes the elbows readable top to bottom.
    """
    order = []

    def walk(tag):
        order.append(tag)
        for kid in sorted(children[tag], key=lambda t: (by_tag[t]["iteration"], by_tag[t]["index"])):
            walk(kid)

    for root in sorted(roots, key=lambda t: by_tag[t]["index"]):
        walk(root)
    return {tag: i for i, tag in enumerate(order)}


def plot_tree(seed, records, out_path, cmap=DEFAULT_CMAP, size_by_fitness=True,
              norm=None):
    """
    The forest as a plain indented family tree, one figure per seed.

    `norm` maps fitness to colour and to marker area. Pass the one built over
    every seed, otherwise each file gets its own scale and the darkest, biggest
    node of a weak seed looks exactly like the darkest node of a strong one.
    """
    by_tag, children, roots = build_forest(records)
    rows = layout_tree(by_tag, children, roots)
    scored = [r["fitness"] for r in by_tag.values() if r["fitness"] is not None]
    if norm is None:
        norm = LogNorm(vmin=min(scored), vmax=max(scored))
    fit_cmap = fitness_cmap(cmap)
    lo, hi = SIZE_RANGE

    def marker_size(fitness):
        if not size_by_fitness:
            return 95
        return lo + (hi - lo) * float(norm(fitness))

    best = best_record(by_tag)
    iters = sorted({r["iteration"] for r in by_tag.values()})

    fig_h = 0.21 * len(rows) + 1.8
    fig, ax = plt.subplots(figsize=(9.5, fig_h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Elbow connectors: down the parent's column, then across to the child.
    for tag, kids in children.items():
        px, py = by_tag[tag]["iteration"], rows[tag]
        for kid in kids:
            kx, ky = by_tag[kid]["iteration"], rows[kid]
            ax.plot([px + 0.18, px + 0.18, kx - 0.16], [py + 0.22, ky, ky],
                    color=EDGE, linewidth=1.0, solid_capstyle="round", zorder=1)

    labels = []
    for tag, r in by_tag.items():
        x, y = r["iteration"], rows[tag]
        failed = r["fitness"] is None
        color = MUTED if failed else fit_cmap(norm(r["fitness"]))
        ax.scatter([x], [y], s=70 if failed else marker_size(r["fitness"]),
                   color=color, zorder=3, marker="X" if failed else "o",
                   edgecolors=ORANGE if best and tag == best["tag"] else SURFACE,
                   linewidths=2.0 if best and tag == best["tag"] else 1.2)
        label = f"{short(tag)}  failed" if failed else f"{short(tag)}  {r['fitness']:.2f}"
        vals = trial_values(r)
        if vals and not failed:
            # The scores the node is the IQM of, so the row shows both what the
            # candidate was ranked on and how far apart its trainings were.
            label += "  [" + " ".join(f"{v:.2f}" for v in vals) + "]"
        labels.append(label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(11, 0),
                    va="center", fontsize=7.5,
                    color=TEXT_SECONDARY if failed else TEXT_PRIMARY)

    # Right margin for the longest row. A label is drawn in points and the axis
    # is measured in iterations, so the padding needed depends on both: listing
    # the repeats roughly triples the text, while a longer run makes each
    # iteration narrower. Solving `pad_in_points >= widest label` for the pad
    # keeps every row inside the axes at any run length and any `repeats`.
    k = min(0.55, (LABEL_CHAR_PT * max(len(t) for t in labels) + 14) / AXES_PT)
    ax.set_xlim(iters[0] - 0.4,
                iters[-1] + k * (0.4 + iters[-1] - iters[0]) / (1 - k))
    ax.set_ylim(len(rows) - 0.5, -0.8)   # row 0 on top
    ax.set_xticks(iters)
    ax.set_xlabel("iteration", fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks([])
    encoding = "colour and size: fitness" if size_by_fitness else "colour: fitness"
    has_reps = any(trial_values(r) for r in by_tag.values())
    ax.set_title(f"seed {seed} -- reward candidate family tree"
                 + (" (lineage reconstructed from the per-iteration winners)"
                    if is_inferred(by_tag) else "") + "\n"
                 f"label: iteration.index, fitness"
                 + (", [every repeat]" if has_reps else "") + f"; {encoding}, "
                 f"on the same scale for every seed\n"
                 + ("fitness is the IQM of the repeats; " if has_reps else "")
                 + "grey X: every training failed; orange ring: run's best",
                 fontsize=11, color=TEXT_PRIMARY, loc="left", pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=fit_cmap), ax=ax,
                      fraction=0.03, pad=0.02, shrink=min(1.0, 26 / len(rows)),
                      anchor=(0.0, 1.0))
    cb.set_label("fitness (log)", fontsize=9, color=TEXT_SECONDARY)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def shared_scale(histories):
    """
    One fitness scale for every seed: `(floor, ylim, norm)`.

    Called once on all loaded histories. `floor` is the dashed line failed jobs
    sit on, `ylim` the y range of every fitness panel, `norm` the fitness ->
    colour/size mapping of every tree. Sharing them is what makes a low seed
    look low next to a high one.
    """
    runs = [r for recs in histories.values() for r in recs if r["phase"] == "run"]
    fits = [r["fitness"] for r in runs if r["fitness"] is not None]
    if not fits:
        return 0.01, None, None
    lo, hi = min(fits), max(fits)
    # The y range has to hold the repeat bars as well, not just the aggregates.
    # An IQM never reaches its candidate's luckiest training, so a range built
    # from the aggregates alone would cut the top off every bar.
    spread = [v for r in runs for v in trial_values(r)]
    floor = min([lo] + spread) * 0.45
    # `norm` stays on the aggregates: it colours and sizes the tree nodes, and
    # those are candidates, so their scale is the scale of candidate fitness.
    return floor, (floor * 0.7, max([hi] + spread) * 2.2), LogNorm(vmin=lo, vmax=hi)


def repeat_line(by_tag):
    """
    One line on what the repeats bought, or None for a history without them.

    Two numbers are worth knowing. How far apart a candidate's own trainings
    landed (the median best/worst ratio, so 1.00 would mean the RL seed made no
    difference at all). And whether ranking on the luckiest training instead of
    the IQM would have handed the run to a different candidate: when it would,
    the trimming changed which reward the search kept, not just the number
    printed next to it.
    """
    measured = [(t, r["fitness"], trial_values(r)) for t, r in by_tag.items()
                if r["fitness"] is not None and len(trial_values(r)) >= 2]
    if not measured:
        return None
    spreads = sorted(max(v) / min(v) for _, _, v in measured if min(v) > 0)
    top_iqm = max(measured, key=lambda m: m[1])[0]
    top_luck = max(measured, key=lambda m: max(m[2]))[0]
    line = (f"  repeats: {len(measured)} candidates measured up to "
            f"{max(len(v) for _, _, v in measured)}x")
    if spreads:
        line += f", median best/worst {spreads[len(spreads) // 2]:.2f}x"
    return line + ("; the luckiest training agrees" if top_iqm == top_luck else
                   f"; ranking on the luckiest training would have crowned "
                   f"{short(top_luck)} instead of {short(top_iqm)}")


def report(seed, records):
    """Text summary of the genealogy, printed next to the figures."""
    by_tag, children, roots = build_forest(records)
    best = best_record(by_tag)
    fertile = sorted(((len(k), t) for t, k in children.items() if k), reverse=True)
    survivors = [r["tag"] for r in by_tag.values() if r.get("survived")]
    lines = [
        f"seed {seed}: {len(by_tag)} candidates over {max(r['iteration'] for r in by_tag.values())} "
        f"iterations, {len(roots)} roots, {len(fertile)} of them ever became a parent",
        f"  best: {best['tag']} (fitness {best['fitness']:.4f}), "
        f"ancestry {' -> '.join(short(t) for t in ancestry(by_tag, best['tag']))}"
        if best else "  best: none scored",
        "  most children: " + ", ".join(f"{short(t)}x{n}" for n, t in fertile[:5]),
        # The greedy runs had no pool: one winner per iteration carried the whole
        # conversation, so the parents themselves are the equivalent line.
        ("  final survivor pool: " + ", ".join(sorted(short(t) for t in survivors))
         if survivors else
         "  per-iteration winners (the greedy chain): "
         + " -> ".join(short(t) for _, t in
                       sorted((by_tag[t]["iteration"], t) for t in children if children[t]))),
    ]
    repeats = repeat_line(by_tag)
    if repeats:
        lines.insert(2, repeats)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "_analysis"))
    p.add_argument("--cmap", default=DEFAULT_CMAP,
                   help="matplotlib colormap for the tree nodes (default: %(default)s)")
    p.add_argument("--no-size", action="store_true",
                   help="draw every tree node the same size (colour only)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = {}
    for seed in args.seeds:
        recs = load_history(args.runs_dir, seed, args.task)
        if recs:
            histories[seed] = recs
    if not histories:
        raise SystemExit("no histories found")
    floor, ylim, norm = shared_scale(histories)

    ncols = 2
    nrows = int(np.ceil(len(histories) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0 * ncols, 4.6 * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.set_visible(False)
    for ax, (seed, recs) in zip(axes.flat, histories.items()):
        ax.set_visible(True)
        plot_fitness_lineage(ax, seed, recs, floor=floor, ylim=ylim)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8,
               markeredgecolor=SURFACE, label="candidate that produced children (a survivor)"),
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=6,
               markeredgecolor=SURFACE, label="dead end (no children)"),
        Line2D([], [], marker="x", linestyle="", color=MUTED, markersize=7,
               label="every training failed (no fitness)"),
        Line2D([], [], color=EDGE, linewidth=1.2, label="parent -> child"),
        Line2D([], [], marker="*", linestyle="-", color=ORANGE, markersize=13,
               markeredgecolor=SURFACE, label="run's best candidate and its ancestry"),
    ]
    # Only claim the repeat bars when something on the page actually has them:
    # a pre-`repeats` history draws none, and a legend entry for a mark that is
    # not in the figure is worse than no entry.
    if any(trial_values(r) for recs in histories.values() for r in recs
           if r["phase"] == "run"):
        handles.insert(3, Line2D([], [], color=EDGE, linewidth=4, alpha=0.75,
                                 solid_capstyle="round",
                                 label="spread of the repeats (dot = their IQM)"))
    # Title band measured in inches, not in figure fractions. The figure grows
    # with the number of seeds (10 seeds = 5 rows = 23in tall), and a fixed
    # fraction of that shrinks to nothing, which used to drop the legend on top
    # of the title. Inches keep the same band whatever the seed count.
    fig_h = fig.get_figheight()
    top = 1 - 1.15 / fig_h
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=9, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0.5, 1 - 0.62 / fig_h))
    fig.suptitle("Reward-candidate lineage per seed: every training and the parent it came from",
                 fontsize=14, color=TEXT_PRIMARY, x=0.006, ha="left",
                 y=1 - 0.18 / fig_h)
    fig.tight_layout(rect=[0, 0, 1, top])
    out = out_dir / "lineage_fitness.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info(f"wrote {out}")

    for seed, recs in histories.items():
        plot_tree(seed, recs, out_dir / f"lineage_tree_seed_{seed}.png",
                  cmap=args.cmap, size_by_fitness=not args.no_size, norm=norm)

    print()
    for seed, recs in histories.items():
        print(report(seed, recs))


if __name__ == "__main__":
    main()
