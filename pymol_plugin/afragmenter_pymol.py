from __future__ import annotations

import os
import csv
from typing import Dict, List

from pymol import cmd


# Palette of distinct colors for positive domains (cycled if needed)
DOMAIN_COLORS: List[str] = [
    "tv_blue", "marine", "orange", "forest", "magenta", "yellow",
    "cyan", "salmon", "wheat", "lime", "violet", "hotpink",
    "deepteal", "slate", "tv_green", "tv_red", "tv_yellow",
]


def load_AFragmenter(name: str, base_dir: str = "."):
    """
    Load an AFragmenter result into PyMOL and color its domains.

    Parameters
    ----------
    name : str
        Base name of the AFragmenter result, e.g. 'AF-A0A0H3WBE8-F1-model_v6'.
        The script expects:
            {base_dir}/{name}/{name}.pdb
            {base_dir}/{name}/{name}.csv
    base_dir : str, optional
        Directory that contains the per-ID subdirectories (default: current dir).

    Usage in PyMOL
    --------------
        run /path/to/this_script.py
        load_AFragmenter AF-A0A0H3WBE8-F1-model_v6, base_dir=/path/to/data/out
    """
    obj_name = name  # PyMOL object and selection prefix
    pdb_path = os.path.join(base_dir, name, f"{name}.pdb")
    csv_path = os.path.join(base_dir, name, f"{name}.csv")

    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load structure
    cmd.load(pdb_path, obj_name)

    # Color whole object grey90 first
    cmd.color("grey90", obj_name)

    # Read domain definitions from CSV
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # Map from positive domain ID to assigned color index
    domain_color_map: Dict[int, str] = {}
    next_color_idx = 0

    for row in rows:
        # Expect columns: structure, domain, n_residues, resi_start, resi_end, mean_pae, sequence
        dom_str = row.get("domain", "").strip()
        if not dom_str:
            continue

        try:
            dom_id = int(dom_str)
        except ValueError:
            # Skip non-integer domain labels
            continue

        # Parse residue range (1-based in CSV)
        try:
            resi_start = int(row.get("resi_start", "").strip())
            resi_end = int(row.get("resi_end", "").strip())
        except ValueError:
            continue

        if resi_start > resi_end:
            # Swap if accidentally reversed
            resi_start, resi_end = resi_end, resi_start

        # Build atom selection string for this domain
        sel_expr = f"{obj_name} and resi {resi_start}-{resi_end}"

        # Only recolor positive domains; negatives remain grey90
        if dom_id > 0:
            if dom_id not in domain_color_map:
                # Pick next color in palette
                color = DOMAIN_COLORS[next_color_idx % len(DOMAIN_COLORS)]
                domain_color_map[dom_id] = color
                next_color_idx += 1
            color = domain_color_map[dom_id]
            # Color the atoms directly, without creating a named selection
            cmd.color(color, sel_expr)


def load_AFragmenter_all(base_dir: str = "."):
    """
    Load and color all AFragmenter results found under base_dir.

    For each subdirectory 'name' under base_dir that contains both:
        {base_dir}/{name}/{name}.pdb
        {base_dir}/{name}/{name}.csv
    this will call load_AFragmenter(name, base_dir=base_dir).
    """
    # Enumerate immediate subdirectories of base_dir
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if not os.path.isdir(subdir):
            continue

        pdb_path = os.path.join(subdir, f"{entry}.pdb")
        csv_path = os.path.join(subdir, f"{entry}.csv")
        if os.path.isfile(pdb_path) and os.path.isfile(csv_path):
            print(f"Loading AFragmenter result: {entry} from {base_dir}")
            load_AFragmenter(entry, base_dir=base_dir)


# Register as PyMOL commands so you can call:
#   load_AFragmenter name, base_dir=...
#   load_AFragmenter_all base_dir=...
cmd.extend("load_AFragmenter", load_AFragmenter)
cmd.extend("load_AFragmenter_all", load_AFragmenter_all)