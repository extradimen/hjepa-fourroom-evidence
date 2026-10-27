"""Convert native FourRoom first-hit records to the frozen P7 task schema."""

import math


def task_rows(metrics, *, arm, training_seed, checkpoint_sha256, episodes, reset_seed_start, cap=420):
    successes = metrics["episode_successes"]
    steps = metrics["steps_to_success"]
    seeds = metrics["seeds"]
    if len(successes) != len(episodes) or len(steps) != len(episodes):
        raise ValueError("Native task cardinality mismatch")
    if seeds is not None and len(seeds) != len(episodes):
        raise ValueError("Native seed cardinality mismatch")
    rows = []
    for index, episode in enumerate(episodes):
        expected_seed = reset_seed_start + int(episode)
        if seeds is not None and int(seeds[index]) != expected_seed:
            raise ValueError("Native reset seed mismatch")
        native_step = float(steps[index])
        hit = math.isfinite(native_step)
        if hit != bool(successes[index]):
            raise ValueError("Native success and first-hit disagree")
        if hit and (not native_step.is_integer() or not 1 <= native_step <= cap):
            raise ValueError("First hit outside frozen physical-step cap")
        rows.append({
            "arm": arm,
            "training_seed": training_seed,
            "episode_index": int(episode),
            "reset_seed": expected_seed,
            "first_hit_step_or_null": int(native_step) if hit else None,
            "hit_by_420": hit,
            "termination_reason": "target_reached" if hit else "right_censored_at_420",
            "checkpoint_sha256": checkpoint_sha256,
        })
    return rows
