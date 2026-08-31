#!/usr/bin/env python3
"""
Zip up a run folder for sharing, leaving the model checkpoints behind.

An ARD run directory is dominated by rl_games `.pth` checkpoints (~80% of a task
folder's bytes) that nobody reading the metrics needs. This packages everything
else — `status.json`, the hydra configs, `params/*.yaml`, container logs and the
TensorBoard event files — into a single deflated zip.

Works at any level of `runs/`: point it at a whole task folder or at one
attempt directory, the walk is the same. By default the zip is written into the
run folder itself (`iter4_run_6/iter4_run_6.zip`); `-o` overrides that.

Usage:
    python scripts/get_run_metrics.py runs/Isaac-ARD-Cartpole-v0
    python scripts/get_run_metrics.py runs/Isaac-ARD-Cartpole-v0/iter4_run_6 -o /tmp/run6.zip
    python scripts/get_run_metrics.py runs/Isaac-ARD-Cartpole-v0 --no-tfevents
    python scripts/get_run_metrics.py runs/Isaac-ARD-Cartpole-v0 --dry-run
"""

import os
import sys
import argparse
import fnmatch
import zipfile
from pathlib import Path

# Always dropped: model weights (the whole point), build noise, and any archive
# a previous packaging left behind (the default output sits inside the run folder,
# so zipping a parent would otherwise re-zip its children's archives).
ALWAYS_EXCLUDE_FILES = ["*.pth", "*.pyc", "*.zip"]
ALWAYS_EXCLUDE_DIRS = ["__pycache__"]

TFEVENTS_GLOB = "events.out.tfevents.*"


def human(n_bytes):
    """Format a byte count for the summary."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} B"
        n_bytes /= 1024


def classify_skip(name):
    """Bucket a skipped file for the summary breakdown."""
    if fnmatch.fnmatch(name, "*.pth"):
        return "pth"
    if fnmatch.fnmatch(name, TFEVENTS_GLOB):
        return "tfevents"
    return "other"


def collect(run_dir, exclude_files, output_path=None):
    """Walk run_dir, returning (included, skipped) lists of (path, arcname, size).

    Excluded directories are pruned in place so their subtrees are never stat'd.
    Symlinks are skipped rather than followed, so the archive can't escape the
    tree or duplicate content. ``output_path`` is skipped outright, since the
    default output sits inside run_dir and an earlier archive left there must not
    be swept into the new one.
    """
    root_name = run_dir.name
    included, skipped = [], []

    for dirpath, dirnames, filenames in os.walk(run_dir):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in ALWAYS_EXCLUDE_DIRS
            and not os.path.islink(os.path.join(dirpath, d))
        )

        for filename in sorted(filenames):
            full = Path(dirpath) / filename
            if full.is_symlink() or full == output_path:
                continue
            arcname = os.path.join(root_name, str(full.relative_to(run_dir)))
            try:
                size = full.stat().st_size
            except OSError:
                continue  # vanished mid-walk (a run still writing)

            if any(fnmatch.fnmatch(filename, pat) for pat in exclude_files) or \
               any(fnmatch.fnmatch(arcname, pat) for pat in exclude_files):
                skipped.append((full, arcname, size))
            else:
                included.append((full, arcname, size))

    return included, skipped


def resolve_output(run_dir, output, force):
    """Work out where the zip goes, refusing to clobber without --force."""
    default_name = f"{run_dir.name}.zip"
    if output is None:
        # Default: sit alongside the metrics it came from, inside the run folder.
        out = run_dir / default_name
    else:
        out = Path(output)
        if out.is_dir():
            out = out / default_name
        elif out.suffix != ".zip":
            out = out.with_suffix(".zip")

    if out.exists() and not force:
        sys.exit(f"error: {out} already exists (use --force to overwrite)")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Zip a run folder for sharing, excluding .pth checkpoints."
    )
    parser.add_argument("run_dir", type=str,
                        help="run folder to package (a task folder or a single run dir)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output zip path or directory "
                             "(default: <run_dir>/<run_dir name>.zip)")
    parser.add_argument("--no-tfevents", action="store_true",
                        help="also drop TensorBoard event files (leaves configs and logs only)")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                        help="extra glob to exclude; repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be packaged without writing the zip")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"error: {run_dir} does not exist")
    if not run_dir.is_dir():
        sys.exit(f"error: {run_dir} is not a directory")

    exclude_files = list(ALWAYS_EXCLUDE_FILES) + list(args.exclude)
    if args.no_tfevents:
        exclude_files.append(TFEVENTS_GLOB)

    out = resolve_output(run_dir, args.output, args.force)

    print(f"Scanning {run_dir} ...")
    included, skipped = collect(run_dir, exclude_files, output_path=out)

    included_bytes = sum(s for _, _, s in included)
    skipped_totals = {"pth": [0, 0], "tfevents": [0, 0], "other": [0, 0]}
    for _, arcname, size in skipped:
        bucket = skipped_totals[classify_skip(os.path.basename(arcname))]
        bucket[0] += 1
        bucket[1] += size

    if not included:
        print(f"warning: no files to package under {run_dir}")

    if not args.dry_run:
        # Progress by top-level child, so a multi-GB walk isn't silent.
        written = 0
        last_group = None
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as zf:
                for full, arcname, _ in included:
                    parts = Path(arcname).parts
                    group = parts[1] if len(parts) > 2 else None
                    if group is not None and group != last_group:
                        print(f"  + {group}")
                        last_group = group
                    zf.write(full, arcname)
                    written += 1
        except KeyboardInterrupt:
            out.unlink(missing_ok=True)
            sys.exit("\naborted; partial archive removed")

    total_bytes = included_bytes + sum(b[1] for b in skipped_totals.values())
    print(f"\n{'Would package' if args.dry_run else 'Packaged'} {run_dir}")
    print(f"  included : {len(included)} files, {human(included_bytes)}")
    for label, (count, size) in skipped_totals.items():
        if count:
            print(f"  skipped  : {count} {label} ({human(size)})")

    if args.dry_run:
        print(f"  archive  : {out} (not written, --dry-run)")
    else:
        zip_bytes = out.stat().st_size
        pct = (zip_bytes / total_bytes * 100) if total_bytes else 0.0
        print(f"  archive  : {out}  {human(zip_bytes)}  ({pct:.1f}% of source)")


if __name__ == "__main__":
    main()
