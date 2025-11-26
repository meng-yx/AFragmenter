
import os
import numpy as np
import pandas as pd

from afragmenter.intervals import calculate_cluster_average_pae
from afragmenter import AFragmenter, fetch_afdb_data

#---------- Function to convert result to domain DataFrame ----------
def result_to_domain_df(result, include_disordered: bool = True):
    """
    Build a DataFrame with columns:
      - domain      : positive ints for domains, negative for non-domain regions
      - n_residues  : number of residues in the region
      - resi_start  : 1-based start residue index (inclusive)
      - resi_end    : 1-based end residue index (inclusive)
      - mean_pae    : average PAE over all residue pairs in the region
      - sequence    : amino acid sequence for the region

    Uses the full sequence from result.sequence_reader.sequence.
    Assumes result.cluster_intervals uses 0-based [start, end] indices (inclusive).
    """
    if result.sequence_reader is None or not hasattr(result.sequence_reader, "sequence"):
        raise ValueError(
            "result.sequence_reader.sequence is not available. "
            "Make sure you constructed AFragmenter with a sequence_file."
        )

    full_seq = result.sequence_reader.sequence
    pae = result.pae_matrix
    n_res = pae.shape[0]
    intervals = result.cluster_intervals  # {cluster_id: [(start, end), ...]} 0-based
    rows = []

    def min_start(segs):
        return min(s for s, _ in segs)

    # Domains (positive ids)
    for cid, segs in intervals.items():
        resi_start0 = min_start(segs)
        resi_end0 = max(e for _, e in segs)
        n_residues = sum(e - s + 1 for s, e in segs)
        mean_pae = float(calculate_cluster_average_pae(pae, segs))

        # 1-based indices
        resi_start = resi_start0 + 1
        resi_end = resi_end0 + 1

        # Concatenate potentially multiple segments in sequence order
        segs_sorted = sorted(segs, key=lambda x: x[0])
        seq_segments = [full_seq[s:e+1] for s, e in segs_sorted]
        seq_region = "".join(seq_segments)

        rows.append({
            "domain": cid + 1,
            "n_residues": n_residues,
            "resi_start": resi_start,
            "resi_end": resi_end,
            "mean_pae": mean_pae,
            "sequence": seq_region,
            "_start": resi_start0,
        })

    if include_disordered:
        covered = np.zeros(n_res, dtype=bool)
        for segs in intervals.values():
            for s, e in segs:
                covered[s:e+1] = True

        neg_id = -1
        i = 0
        while i < n_res:
            if covered[i]:
                i += 1
                continue
            start = i
            while i < n_res and not covered[i]:
                i += 1
            end = i - 1
            segs = [(start, end)]
            n_residues = end - start + 1
            mean_pae = float(calculate_cluster_average_pae(pae, segs))

            resi_start = start + 1
            resi_end = end + 1
            seq_region = full_seq[start:end+1]

            rows.append({
                "domain": neg_id,
                "n_residues": n_residues,
                "resi_start": resi_start,
                "resi_end": resi_end,
                "mean_pae": mean_pae,
                "sequence": seq_region,
                "_start": start,
            })
            neg_id -= 1

    df = pd.DataFrame(rows).sort_values("_start").drop(columns=["_start"]).reset_index(drop=True)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AFragmenter on AFDB PAE matrix and output domains to CSV and structure to PDB."
    )
    parser.add_argument(
        "--id", required=True,
        help="UniProt/AlphaFoldDB identifier (e.g. 'Q8IZT6') or path to a text file with one ID per line."
    )
    parser.add_argument(
        "--out_root", required=True,
        help="Root output directory for results."
    )
    parser.add_argument(
        "--min_size", type=int, default=30,
        help="Minimum size of a domain (default: 30)."
    )
    parser.add_argument(
        "--min_avg_pae", type=float, default=15,
        help="Minimum average PAE of a domain (default: 15)."
    )
    parser.add_argument(
        "--max_overlap", type=int, default=40,
        help="Maximum overlap between domains (default: 40)."
    )
    parser.add_argument(
        "--collapse_intervals", action="store_true", default=True,
        help="Collapse intervals to prevent small domains being enclosed by larger domains. (default: True)"
    )
    parser.add_argument(
        "--no-collapse_intervals", dest="collapse_intervals", action="store_false",
        help="Do not collapse intervals (overrides --collapse_intervals)."
    )
    parser.add_argument(
        "--resolution", type=float, default=0.4,
        help="Resolution for the PAE matrix (default: 0.4)."
    )

    args = parser.parse_args()

    # Determine whether --id is a single ID or a path to a file containing IDs
    if os.path.isfile(args.id):
        with open(args.id, "r") as fh:
            id_list = [line.strip() for line in fh if line.strip()]
    else:
        id_list = [args.id]

    # ------------ workflow starts here ------------
    for uniprot_id in id_list:
        print(f"Fetching AFDB data for {uniprot_id}...")
        try:
            pae, structure, structure_basename = fetch_afdb_data(uniprot_id, structure_format='pdb')
        except Exception as e:
            print(f"Error fetching AFDB data for {uniprot_id}: {e}")
            continue

        # Run AFragmenter
        print(f"Fragmenting {uniprot_id}...")
        try:
            fragmenter = AFragmenter(pae, sequence_file=structure)
            result = fragmenter.cluster(
                resolution=args.resolution,
                objective_function="modularity",
                min_size=args.min_size,
                min_avg_pae=args.min_avg_pae,
                collapse_intervals=args.collapse_intervals,
                max_overlap=args.max_overlap,
            )
        except Exception as e:
            print(f"Error fragmenting {uniprot_id}: {e}")
            continue

        print(f"Writing out files for {uniprot_id}...")
        try:
            # Convert result to domain DataFrame
            df_result = result_to_domain_df(result)

            # Insert structure_basename as the first column
            df_result.insert(0, 'structure', structure_basename)

            # Save results
            model_id = structure_basename.replace('.pdb', '')
            outdir = os.path.join(args.out_root, model_id)
            os.makedirs(outdir, exist_ok=True)

            # Save domain DataFrame
            df_result.to_csv(os.path.join(outdir, f'{model_id}.csv'), index=False)

            # Save .pdb structure
            with open(os.path.join(outdir, structure_basename), 'w') as f:
                f.write(structure)
        except Exception as e:
            print(f"Error writing out files for {uniprot_id}: {e}")
            continue

if __name__ == "__main__":
    main()
