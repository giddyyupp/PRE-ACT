import hashlib
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence


def get_fps_mmau(video_id, dataset_type="CAP"):
    assert dataset_type in ["CAP", "DADA"]  # 仅支持 'CAP' 和 'DADA' 数据集
    if dataset_type == "CAP":
        if "000001" <= video_id <= "006381":
            return 10
        elif "006382" <= video_id <= "007887":
            return 30
        elif "007888" <= video_id <= "009046":
            return 20
        elif "009047" <= video_id <= "011770":
            return 30
        elif "013001" <= video_id <= "014490":
            return 10
        else:
            return None
    else:  # dataset_type == 'DADA'
        return 30


def _stable_score(video_id: str, seed: int) -> int:
    """
    Stable pseudo-random score.

    Unlike Python's built-in hash(), this is deterministic across
    machines, processes, Python versions, and runs.
    """
    key = f"{seed}:{video_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16)


def deterministic_record_subsets(
    records: Sequence[dict],
    fractions=(0.01, 0.05, 0.10),
    seed: int = 42,
    stratify_by_label: bool = False,
    id_key: str = "video_hashcode",
    label_key: Optional[str] = None,
    label_fn: Optional[Callable[[dict], int]] = None,
) -> Dict[float, List[dict]]:
    """
    Deterministically select nested fractions of videos.

    Guarantees:
        subset[1%] ⊂ subset[5%] ⊂ subset[10%]

    Parameters
    ----------
    records:
        List of per-video dictionaries.

    fractions:
        Fractions of videos to select.

    seed:
        Controls the deterministic ranking.

    stratify_by_label:
        False:
            Ignore labels and sample from all videos.

        True:
            Select the requested fraction independently from
            each label group.

    id_key:
        Record field containing the unique video identifier.

    label_key:
        Record field containing the class label.
        Example: "label" for Nexar.

    label_fn:
        Optional function extracting the label from a record.

        Example for DAD:
            lambda r: int(r["is_positive"])

        If provided, label_fn takes precedence over label_key.
    """

    records = list(records)

    if len(records) == 0:
        raise ValueError("records is empty.")

    # ---------------------------------------------------------
    # Check IDs
    # ---------------------------------------------------------
    video_ids = []

    for r in records:
        if id_key not in r:
            raise KeyError(
                f"Record does not contain id_key='{id_key}': {r}"
            )

        video_ids.append(str(r[id_key]))

    if len(video_ids) != len(set(video_ids)):
        raise ValueError(
            f"{id_key} must uniquely identify every video."
        )

    # Convenient lookup
    record_by_id = {
        str(r[id_key]): r
        for r in records
    }

    # =========================================================
    # NON-STRATIFIED
    # =========================================================
    if not stratify_by_label:

        ranked_ids = sorted(
            video_ids,
            key=lambda x: _stable_score(x, seed),
        )

        subsets = {}

        for frac in sorted(fractions):

            print(
                f"[SAMPLER DEBUG] total={len(ranked_ids)} "
                f"frac={frac} "
                f"raw={len(ranked_ids) * frac} "
                f"n={round(len(ranked_ids) * frac)}"
            )

            n = max(
                1,
                round(len(ranked_ids) * frac)
            )

            selected_ids = ranked_ids[:n]

            subsets[frac] = [
                record_by_id[x]
                for x in selected_ids
            ]

            print(
                f"[SAMPLER DEBUG] selected={len(subsets[frac])}"
            )

        return subsets

    # =========================================================
    # STRATIFIED
    # =========================================================

    if label_fn is None and label_key is None:
        raise ValueError(
            "For stratified sampling, provide either "
            "label_key or label_fn."
        )

    def get_label(record):

        if label_fn is not None:
            return label_fn(record)

        if label_key not in record:
            raise KeyError(
                f"Record does not contain label_key='{label_key}'."
            )

        return record[label_key]

    # Group by label
    by_label = defaultdict(list)

    for r in records:
        label = get_label(r)
        by_label[label].append(
            str(r[id_key])
        )

    # Deterministic ranking inside each label group
    ranked_by_label = {}

    for label, ids in by_label.items():

        ranked_by_label[label] = sorted(
            ids,
            key=lambda x: _stable_score(x, seed),
        )

    # Build nested subsets
    subsets = {}

    for frac in sorted(fractions):

        selected_ids = []

        for label, ranked_ids in ranked_by_label.items():

            n = max(
                1,
                round(len(ranked_ids) * frac)
            )

            selected_ids.extend(
                ranked_ids[:n]
            )

        # Give combined subset deterministic ordering too
        selected_ids = sorted(
            selected_ids,
            key=lambda x: _stable_score(x, seed),
        )

        subsets[frac] = [
            record_by_id[x]
            for x in selected_ids
        ]

    return subsets


def print_subset_stats(
    subsets,
    label_fn=None,
):
    for frac, subset in sorted(subsets.items()):

        print(
            f"{100 * frac:5.1f}%: "
            f"{len(subset):5d} videos",
            end=""
        )

        if label_fn is not None:

            counts = defaultdict(int)

            for r in subset:
                counts[label_fn(r)] += 1

            print(
                " | "
                + ", ".join(
                    f"label={k}: {v}"
                    for k, v in sorted(counts.items())
                )
            )

        else:
            print()

