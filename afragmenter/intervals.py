import igraph
import numpy as np

from typing import Optional, Union

def find_cluster_intervals(clusters: igraph.VertexClustering,
                           collapse_intervals: bool = False) -> dict:
    """
    Create a dictionary of cluster intervals from the vertex clustering object.

    Parameters:
    - clusters (igraph.VertexClustering): The vertex clustering object.
    - collapse_intervals (bool, optional): If True, represent each cluster with a single interval spanning
                                           from the minimum to maximum residue index in that cluster, even
                                           if the residues are discontinuous. Defaults to False.

    Returns:
    - dict: A dictionary where the keys are the cluster indices and the values are lists of tuples representing the cluster intervals.

    Example output:
    {
        0: [(0, 20), (35, 40)],
        1: [(21, 34)]
    }
    """
    results = {}
    for i, cluster in enumerate(clusters):
        region = []
        if cluster:
            if collapse_intervals:
                # Collapse all residues in this cluster into a single interval [min, max]
                start = min(cluster)
                end = max(cluster)
                region.append((start, end))
            else:
                # Preserve the original behavior: split into contiguous intervals
                start = cluster[0]
                for j in range(1, len(cluster)):
                    if cluster[j] != cluster[j - 1] + 1:
                        region.append((start, cluster[j - 1]))
                        start = cluster[j]
                region.append((start, cluster[-1]))
        results[i] = region
    return results



def remove_min_size_intervals(intervals: dict, min_size: int) -> dict:
    """
    Remove clusters that are smaller than a given size based on the *total* size of the intervals (values) per group (keys).

    In example 1, clusters 0, 2 and 4 are removed because the intervals are too small.
    In example 2, clusters 0 and 3 are removed because the intervals are too small. Cluser 1 has intervals that
    are smaller than the min_size threshold, but the total size of the intervals is larger than min_size, so it is kept.

    Example 1:
        intervals = {0: [(0, 2)],                      -> total size interval = 3
                    1: [(3, 33)],                      -> total size interval = 31
                    2: [(34, 35)], 
                    3: [(36, 133), (143, 269)],        -> total size interval = 225
                    4: [(134, 138)]}
        min_size = 10
        merge_intervals(intervals, min_size) -> {1: [(3, 33)], 
                                                 3: [(36, 133), (143, 269)]}
    
    Example 2:
        intervals = {0: [(0, 5)],                                               -> total size interval = 6
                    1: [(6, 11), (807, 807), (810, 811), (813, 933)],           -> total size interval = 130
                    2: [(12, 179), (419, 518), (519, 603), (604, 806), 
                        (808, 809), (812, 812)],
                    3: [(180, 182)], 
                    4: [(183, 229), (230, 418)]}
        min_size = 10
        merge_intervals(intervals, min_size) -> {1: [(6, 11), (807, 807), (810, 811), (813, 933)],
                                                 2: [(12, 179), (419, 518), (519, 603), (604, 806), 
                                                     (808, 809), (812, 812)],
                                                 4: [(183, 229), (230, 418)]}

    Parameters:
    - intervals (dict): A dictionary where the keys are the cluster indices and the values are lists of 
                        tuples representing the cluster intervals.
    - min_size (int): The minimum total size of the intervals to keep.

    Returns:
    - dict: A dictionary with the same structure as the input, but with intervals smaller than min_size removed.
    """
    filtered_intervals = {}
    for key, value in intervals.items():
        total_size = sum(interval[1] - interval[0] + 1 for interval in value)
        if total_size >= min_size:
            filtered_intervals[key] = value

    # Remove any empty entries
    filtered_intervals = {k: v for k, v in filtered_intervals.items() if v}
    
    return filtered_intervals


def merge_intervals(intervals: dict, min_size: int) -> dict:
    """
    Iterates over all intervals and removes those that are smaller than the min_size threshold regardless of 
    the total size of the intervals in a cluster/group (only the size of the individual intervals is considered).
    Merges adjacent intervals of the same cluster while skipping intervals below the min_size threshold.

    Example:
        intervals = {0: [(0, 2)], 1: [(3, 33)], 2: [(34, 35)], 3: [(36, 133), (143, 269)], 4: [(134, 138)], 
                    5: [(139, 142)], 6: [(270, 272)]}
        min_size = 10
        merge_intervals(intervals, min_size) -> {1: [(3, 33)], 3: [(36, 269)]}

    In the example, cluster 0, 2 and 4 are removed because the intervals are too small. The intervals of cluster 3 
    are merged because there is no other cluster with a size >= min_size between its intervals.

    Parameters:
    - intervals (dict): A dictionary where the keys are the cluster indices and the values are lists of 
                        tuples representing the cluster intervals.
    - min_size (int): The minimum total size of the intervals to keep.

    Returns:
    - dict: A dictionary with the same structure as the input, but with intervals smaller than min_size 
            removed and the keys renumbered starting
    """
    class Interval:
        def __init__(self, group: int, start: int, end: int):
            self.group = group
            self.start = start
            self.end = end
            self.size = end - start + 1
        
        def __repr__(self):
            return f"Interval(group={self.group}, start={self.start}, end={self.end}, size={self.size})"

    interval_objects = []
    for key, value in intervals.items():
        for results in value:
            start, end = results
            interval_objects.append(Interval(key, start, end))
    interval_objects.sort(key=lambda x: x.start)

    previous_valid = None
    results = []
    for current in interval_objects:
        # Skip intervals that are too small
        if current.size < min_size:
            continue

        # First interval above the min_size threshold
        if previous_valid is None:
            previous_valid = current
            continue

        # If the current interval is adjacent to the previous one and of the same group, merge them
        if previous_valid.group == current.group:
            previous_valid = Interval(previous_valid.group, previous_valid.start, current.end)
        else:
            results.append(previous_valid)
            previous_valid = current
        
    # Add the last interval
    if previous_valid is not None:
        results.append(previous_valid)

    # Group the merged intervals by cluster
    merged_intervals = {}
    for interval in results:
        if interval.group not in merged_intervals:
            merged_intervals[interval.group] = []
        merged_intervals[interval.group].append((interval.start, interval.end))
    
    return merged_intervals


def calculate_cluster_average_pae(pae_matrix: np.ndarray, intervals: list) -> Union[np.float64, float]:
    """
    Calculate the average PAE value within a cluster.
    
    Parameters:
    - pae_matrix (np.ndarray): The Predicted Aligned Error matrix
    - intervals (list): List of (start, end) tuples defining the cluster regions

    Returns:
    - float or np.float64: Average PAE value for intra-cluster residue pairs
    """
    # Collect all residues from all intervals in this cluster
    cluster_residues = []
    for start, end in intervals:
        cluster_residues.extend(range(start, end + 1))
    
    if len(cluster_residues) < 2:
        return 0.0
        
    pae_values = []
    
    # Calculate PAE for all pairs within the cluster (including between segments)
    for i in range(len(cluster_residues)):
        for j in range(i + 1, len(cluster_residues)):
            res_i = cluster_residues[i]
            res_j = cluster_residues[j]
            
            # Make sure indices are within bounds
            if res_i < pae_matrix.shape[0] and res_j < pae_matrix.shape[1]:
                pae_values.append(pae_matrix[res_i][res_j])
                # Only add the symmetric value if the matrix is not symmetric
                if pae_matrix[res_i][res_j] != pae_matrix[res_j][res_i]:
                    pae_values.append(pae_matrix[res_j][res_i])
    
    return np.mean(pae_values) if pae_values else 0.0


def filter_intervals_by_average_pae(intervals: dict, pae_matrix: np.ndarray, 
                               min_avg_pae: float = 5.0, verbose: bool = False) -> dict:
    """
    Filter out cluster intervals that have average PAE values above the threshold.
    
    This function removes clusters where the average PAE between all residue pairs 
    within the cluster exceeds the min_avg_pae threshold. 
    Can helpfilter out poorly structured domains with low confidence.
    
    Parameters:
    - intervals: Dictionary where keys are cluster indices and values are lists of 
                (start, end) tuples representing cluster intervals
    - pae_matrix: The Predicted Aligned Error matrix (2D numpy array)
    - min_avg_pae: Maximum allowed average PAE for a cluster (clusters above this are removed)
    - verbose: Whether to print filtering information
        
    Returns:
    - Dictionary with same structure as input, but with high-PAE clusters removed
        and keys renumbered starting from zero
        
    Example:
        intervals = {0: [(0, 119), (299, 320)], 1: [(120, 298)], 2: [(321, 350)]}
        pae_matrix = np.array(...)  # PAE matrix
        min_avg_pae = 5.0
        
        # If cluster 2 has avg PAE = 7.2, it gets filtered out
        filtered = filter_intervals_by_avg_pae(intervals, pae_matrix, min_avg_pae, verbose=True)
        # Returns: {0: [(0, 119), (299, 320)], 1: [(120, 298)]}
    """
    if min_avg_pae < 0:
        raise ValueError("Minimum average PAE must be greater than or equal to 0")
    
    filtered_intervals = {}
    cluster_pae_stats = {}
    removed_clusters = []
    
    for cluster_id, cluster_intervals in intervals.items():
        avg_pae = calculate_cluster_average_pae(pae_matrix, cluster_intervals)
        cluster_pae_stats[cluster_id] = avg_pae
        
        total_residues = sum(end - start + 1 for start, end in cluster_intervals)
        is_discontinuous = len(cluster_intervals) > 1
        
        if avg_pae <= min_avg_pae:
            filtered_intervals[cluster_id] = cluster_intervals
            if verbose:
                discontinuous_note = " (discontinuous)" if is_discontinuous else ""
                print(f"Cluster {cluster_id}: {total_residues} residues{discontinuous_note}, "
                      f"avg PAE = {avg_pae:.2f} - KEPT")
        else:
            removed_clusters.append((cluster_id, cluster_intervals, avg_pae))
            if verbose:
                discontinuous_note = " (discontinuous)" if is_discontinuous else ""
                print(f"Cluster {cluster_id}: {total_residues} residues{discontinuous_note}, "
                      f"avg PAE = {avg_pae:.2f} - FILTERED OUT")
    
    if verbose:
        print(f"\nPAE filtering summary:")
        print(f"\tOriginal clusters: {len(intervals)}")
        print(f"\tClusters kept: {len(filtered_intervals)}")
        print(f"\tClusters removed: {len(removed_clusters)}")
    
    renumbered_intervals = {i: v for i, (_, v) in enumerate(filtered_intervals.items())}
    
    return renumbered_intervals


def remove_enclosed_clusters(intervals: dict) -> dict:
    """
    Remove clusters whose overall span is fully enclosed by another cluster's span.

    This helper is useful, for example, after clustering with ``collapse_intervals=True``,
    where each cluster is represented by a single (start, end) interval. It also works
    when clusters have multiple intervals: in that case the overall span for a cluster
    is defined as (min(start), max(end)) across all its intervals.

    Any cluster A whose span [start_A, end_A] is fully contained within another cluster B's
    span [start_B, end_B] (with B's span greater than or equal in length) will be removed.

    Parameters
    ----------
    intervals : dict
        Dictionary mapping cluster id -> list of (start, end) tuples.

    Returns
    -------
    dict
        New dictionary with enclosed clusters removed.

    Example
    -------
    >>> intervals = {0: [(0, 11)], 1: [(12, 313)], 2: [(171, 176)]}
    >>> cleaned = remove_enclosed_clusters(intervals)
    >>> cleaned
    {0: [(0, 11)], 1: [(12, 313)]}
    """
    if not intervals:
        return intervals

    # Compute overall span per cluster
    spans = {}
    for cid, ivs in intervals.items():
        if not ivs:
            continue
        starts = [s for (s, e) in ivs]
        ends = [e for (s, e) in ivs]
        spans[cid] = (min(starts), max(ends))

    keep = set(spans.keys())
    for a, (sa, ea) in spans.items():
        if a not in keep:
            continue
        for b, (sb, eb) in spans.items():
            if a == b:
                continue
            # Check if A is fully inside B and B is at least as large
            if sa >= sb and ea <= eb and (eb - sb) >= (ea - sa):
                if a in keep:
                    keep.remove(a)
                break

    return {cid: intervals[cid] for cid in keep}


def _cluster_span(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    """Return overall span (min_start, max_end) for a cluster's intervals."""
    starts = [s for (s, e) in intervals]
    ends = [e for (s, e) in intervals]
    return min(starts), max(ends)


def _cluster_length(intervals: list[tuple[int, int]]) -> int:
    """Return total number of residues in a cluster's intervals."""
    return sum(e - s + 1 for s, e in intervals)


def _weighted_mean_pae_for_candidate(pae_matrix: np.ndarray,
                                     candidate: dict[int, list[tuple[int, int]]]) -> float:
    """
    Compute weighted mean PAE for a small set of clusters defined in `candidate`.
    """
    total_weighted = 0.0
    total_len = 0
    for cid, ivs in candidate.items():
        if not ivs:
            continue
        length = _cluster_length(ivs)
        if length <= 1:
            continue
        avg_pae = calculate_cluster_average_pae(pae_matrix, ivs)
        total_weighted += float(avg_pae) * length
        total_len += length
    if total_len == 0:
        return float("inf")
    return total_weighted / total_len



def resolve_overlaps_by_pae(intervals: dict,
                            pae_matrix: np.ndarray,
                            max_overlap: int) -> dict:
    """
    Resolve partially overlapping domains by minimizing local weighted mean PAE.

    This helper is intended for use after collapsing intervals (collapse_intervals=True),
    where each cluster typically has a single (start, end) interval. It will:

    - Detect pairs of clusters whose spans overlap by > max_overlap residues.
    - For each such pair (A, B), consider three local configurations:
        1. Assign the overlapping region to A only (remove it from B).
        2. Assign the overlapping region to B only (remove it from A).
        3. Merge A and B into a single cluster.
      The configuration with the lowest weighted mean PAE (over the affected clusters)
      is chosen and applied.

    The procedure is greedy: it repeatedly scans for overlapping pairs, resolves the
    first one it finds (if needed), updates the intervals, and restarts the scan until
    no overlaps larger than max_overlap remain.
    """
    if max_overlap < 0:
        raise ValueError("max_overlap must be greater than or equal to 0")
    if not intervals:
        return intervals

    def build_spans(local_intervals: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int, int]]:
        spans_list: list[tuple[int, int, int]] = []
        for cid, ivs in local_intervals.items():
            if not ivs:
                continue
            start, end = _cluster_span(ivs)
            spans_list.append((cid, start, end))
        spans_list.sort(key=lambda x: x[1])
        return spans_list

    def resolve_pair(a_id: int, b_id: int, local_intervals: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
        a_ivs = local_intervals.get(a_id, [])
        b_ivs = local_intervals.get(b_id, [])
        if not a_ivs or not b_ivs:
            return local_intervals

        a_start, a_end = _cluster_span(a_ivs)
        b_start, b_end = _cluster_span(b_ivs)

        ov_start = max(a_start, b_start)
        ov_end = min(a_end, b_end)
        if ov_start > ov_end:
            return local_intervals  # no actual overlap

        # Helper to build non-overlapping pieces for a given interval [s, e]
        def non_overlap_segments(s: int, e: int) -> list[tuple[int, int]]:
            pieces: list[tuple[int, int]] = []
            if s < ov_start:
                pieces.append((s, ov_start - 1))
            if ov_end < e:
                pieces.append((ov_end + 1, e))
            return pieces

        candidates: list[tuple[str, dict[int, list[tuple[int, int]]]]] = []

        # Base copy of current intervals for modification per candidate
        base = {cid: ivs[:] for cid, ivs in local_intervals.items()}

        # Option 1: overlap to A, remove from B
        opt1 = {cid: ivs[:] for cid, ivs in base.items()}
        opt1[a_id] = [(a_start, a_end)]
        b_pieces = non_overlap_segments(b_start, b_end)
        if b_pieces:
            opt1[b_id] = b_pieces
        else:
            opt1.pop(b_id, None)
        candidates.append(("overlap_to_a", opt1))

        # Option 2: overlap to B, remove from A
        opt2 = {cid: ivs[:] for cid, ivs in base.items()}
        opt2[b_id] = [(b_start, b_end)]
        a_pieces = non_overlap_segments(a_start, a_end)
        if a_pieces:
            opt2[a_id] = a_pieces
        else:
            opt2.pop(a_id, None)
        candidates.append(("overlap_to_b", opt2))

        # Option 3: merge A and B into single cluster (use a_id as merged id)
        opt3 = {cid: ivs[:] for cid, ivs in base.items()}
        merged_start = min(a_start, b_start)
        merged_end = max(a_end, b_end)
        opt3[a_id] = [(merged_start, merged_end)]
        opt3.pop(b_id, None)
        candidates.append(("merge", opt3))

        # Evaluate candidates (only over the affected clusters)
        best_candidate = None
        best_score = float("inf")
        for _, cand in candidates:
            local = {cid: cand[cid] for cid in cand if cid in (a_id, b_id)}
            score = _weighted_mean_pae_for_candidate(pae_matrix, local)
            if score < best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            return local_intervals

        return best_candidate

    # Main greedy loop
    while True:
        spans = build_spans(intervals)
        modified = False

        for i in range(len(spans) - 1):
            a_id, a_start, a_end = spans[i]
            b_id, b_start, b_end = spans[i + 1]
            ov_start = max(a_start, b_start)
            ov_end = min(a_end, b_end)
            if ov_start > ov_end:
                continue
            ov_len = ov_end - ov_start + 1
            if ov_len <= max_overlap:
                continue

            # Need to resolve this pair
            intervals = resolve_pair(a_id, b_id, intervals)
            modified = True
            break

        if not modified:
            break

    return intervals


def filter_cluster_intervals(intervals: dict, min_size: int, attempt_merge: bool = True,
                             pae_matrix: Optional[np.ndarray] = None, min_avg_pae: Optional[float] = None) -> dict:
    """
    Filter out intervals that are smaller than a given size. 
    Optionally attempt to merge smaller clusters with adjacent larger ones.

    Parameters:
    - intervals (dict): A dictionary where the keys are the cluster indices and the values are lists of 
                        tuples representing the cluster intervals.
    
    - min_size (int): The minimum total size of the intervals to keep.
    - attempt_merge (bool, optional): Whether to try to merge smaller clusters with adjacent larger ones. 
                                      Defaults to True.

    Returns:
    - dict: A dictionary with the same structure as the input, but with intervals smaller than min_size
            removed and the keys renumbered starting from zero.
    """

    if min_size < 0:
        raise ValueError("Minimum cluster size must be greater than or equal to 0")
    if min_size == 0:
        return intervals

    if attempt_merge:
        intervals = merge_intervals(intervals, min_size)
    else:
        intervals = remove_min_size_intervals(intervals, min_size)

    # Renumber the keys starting from zero
    filtered_intervals = {i: v for i, (_, v) in enumerate(intervals.items())}

    if pae_matrix is not None and min_avg_pae is not None:
        filtered_intervals = filter_intervals_by_average_pae(filtered_intervals, pae_matrix, min_avg_pae)

    return filtered_intervals
