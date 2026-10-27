"""P8 one-time sealed-test FourRoom evaluation, recording first hit through physical step 420."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = next(ROOT.glob("vendor/H-JEPA-*"))
sys.path[:0] = [str(ROOT), str(UPSTREAM / "h_jepa"), str(UPSTREAM)]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def bound(path, digest):
    resolved = (ROOT / path).resolve()
    assert resolved.is_relative_to(ROOT) and sha(resolved) == digest
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = (ROOT / args.job).resolve()
    assert job_path.is_relative_to(ROOT)
    job = json.loads(job_path.read_text())
    assert job["status"] == "frozen" and job["stage"] == "p8_sealed_first_hit"
    assert job["test_access"] is True and job["task_mode"] in ("generate", "reuse")
    protocol = json.loads(bound(job["protocol_path"], job["protocol_sha256"]).read_text())
    recipe = json.loads(bound(job["evaluation_recipe_path"], job["evaluation_recipe_sha256"]).read_text())
    assert recipe["status"] == "frozen" and recipe["physical_action_cap"] == recipe["eval_budget"] == 420
    assert recipe["protocol_sha256"] == job["protocol_sha256"]
    assert recipe["test_access"] is True and protocol["evaluation"]["split"] == "p1_sealed_test"
    assert recipe["sealed_dataset_sha256_from_collection_receipt"] == protocol["evaluation"]["sealed_dataset_sha256_from_collection_receipt"]
    assert job["seed"] in protocol["models"]["seeds"]
    assert job["arm"] in protocol["models"]["arms"]
    assert recipe["episodes"] == protocol["evaluation"]["episodes"]
    authorization = json.loads(bound(job["budget_authorization_path"], job["budget_authorization_sha256"]).read_text())
    assert authorization["status"] == "user_approved" and authorization["cap_seconds"] == job["required_total_cap_seconds"]
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authorization["deadline_utc"])
    project = ROOT.parents[1]
    guard = json.loads((project / "budget-reservations" / f"{job['reservation_id']}.json").read_text())
    assert guard["status"] == "consumed_before_launch" and guard["pid"] == os.getppid()
    training_audit = json.loads(bound(job["training_audit_path"], job["training_audit_sha256"]).read_text())
    assert training_audit["status"] == "payload_audit_passed"
    assert training_audit["arm"] == job["arm"] and training_audit["seed"] == job["seed"]
    assert training_audit["checkpoint_sha256"] == job["checkpoint_sha256"]
    checkpoint_path = bound(job["checkpoint_path"], job["checkpoint_sha256"])
    bn_path = bound(job["bn_path"], job["bn_sha256"])

    assert sha(ROOT / recipe["sealed_dataset_path"]) == recipe["sealed_dataset_sha256_from_collection_receipt"]

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from src.p1_runtime import Runtime
    from src.p7_first_hit import task_rows
    from planning_eval_compat import run_planning_eval
    from planning_eval import load_eval_config

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 4000 and checkpoint["seed"] == job["seed"]
    assert checkpoint["arm"] == job["arm"] and checkpoint["idm"] == job["idm"]
    runtime = Runtime(ROOT)
    for level in (1, 2):
        runtime.cfg[f"level{level}"].idm_coeff = float(job["idm"])
    runtime.model.load_state_dict(checkpoint["model"], strict=True)
    runtime.model.normalizer_artifact = checkpoint["normalizer"]
    del checkpoint
    buffers = torch.load(bn_path, map_location="cpu", weights_only=False)
    state = runtime.model.state_dict()
    expected = {key for key in state if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    assert set(buffers) == expected
    assert all(torch.isfinite(value).all().item() for value in buffers.values() if value.is_floating_point())
    state.update(buffers)
    runtime.model.load_state_dict(state, strict=True)
    runtime.model.eval()
    out = project / "runs" / job["run_name"]
    assert out.parent == project / "runs"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    shared_path = project / "shared-tasksets" / "p8-fourroom-sealed32-420-v1.pt"
    if job["task_mode"] == "generate":
        assert not shared_path.exists()
        ds = swm.data.HDF5Dataset("p1_sealed_test", cache_dir=ROOT / runtime.settings["dataset"],
                                  keys_to_cache=["action"], level1={"frameskip": 1, "num_steps": 1})
        episodes = np.asarray(recipe["episodes"], dtype=np.int64)
        starts = np.full(len(episodes), recipe["start_step"], dtype=np.int64)
        chunks = ds.load_chunk(episodes, starts, starts + recipe["physical_goal_offset"] + 1)
        columns = list(ds.column_names)
        if "seed" not in columns:
            columns.append("seed")
        for index, chunk in enumerate(chunks):
            chunk["seed"] = torch.full((recipe["physical_goal_offset"] + 1,),
                                       recipe["reset_seed_start"] + int(episodes[index]), dtype=torch.int64)
            assert len(chunk["pixels"]) == recipe["physical_goal_offset"] + 1
            assert chunk["distractor_xy"].shape[-1] == 2
        task_payload = {"data": chunks, "columns": columns, "episodes_idx": episodes.tolist(),
                        "start_steps": starts.tolist(), "goal_offset_steps": recipe["physical_goal_offset"],
                        "eval_budget": 420, "process": {}, "split": "sealed_test",
                        "reset_seed_start": recipe["reset_seed_start"],
                        "recipe_sha256": job["evaluation_recipe_sha256"]}
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shared_path.with_suffix(".tmp")
        assert not temporary.exists()
        torch.save(task_payload, temporary)
        temporary.replace(shared_path)
    else:
        assert sha(shared_path) == job["task_payload_sha256"]
    task_sha = sha(shared_path)
    cfg = load_eval_config("fourroom_l2")
    cfg.cache_dir = str(ROOT / runtime.settings["dataset"])
    cfg.policy = "provided_in_memory"
    cfg.load_eval_trajs_path = str(shared_path)
    cfg.dump_eval_trajs_path = None
    cfg.world.min_distractors = recipe["world_correction"]["min_distractors"]
    cfg.world.max_distractors = recipe["world_correction"]["max_distractors"]
    collection_world = OmegaConf.to_container(OmegaConf.load(UPSTREAM / "scripts/data/config/fourroom_tp35_d1.yaml").world, resolve=True)
    planning_world = OmegaConf.to_container(cfg.world, resolve=False)
    for key, value in collection_world.items():
        if key not in ("num_envs", "max_episode_steps", "image_shape"):
            assert planning_world[key] == value, ("Scene mismatch", key)
    cfg.dataset.keys_to_cache = ["action"]
    cfg.eval.dataset_name = "p1_sealed_test"
    cfg.eval.num_eval = len(recipe["episodes"])
    cfg.eval.goal_offset_steps = recipe["physical_goal_offset"]
    cfg.eval.eval_budget = 420
    cfg.eval.chunk_size = 1
    cfg.seed = recipe["planner_seed"]
    for level in ("level1", "level2"):
        cfg.solver.solvers[level].n_steps = recipe["solver_steps"]
        cfg.solver.solvers[level].num_samples = recipe["solver_samples"][level]
    random.seed(recipe["planner_seed"])
    np.random.seed(recipe["planner_seed"])
    torch.manual_seed(recipe["planner_seed"])
    torch.cuda.manual_seed_all(recipe["planner_seed"])
    metrics = run_planning_eval(cfg, model=runtime.model, results_dir=out / "native-results")
    rows = task_rows(metrics, arm=job["arm"], training_seed=job["seed"],
                     checkpoint_sha256=job["checkpoint_sha256"], episodes=recipe["episodes"],
                     reset_seed_start=recipe["reset_seed_start"], cap=420)
    for row in rows:
        row["task_payload_sha256"] = task_sha
    with (out / "first-hit.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    counts = {str(threshold): sum(row["first_hit_step_or_null"] is not None and
                                  row["first_hit_step_or_null"] <= threshold for row in rows)
              for threshold in recipe["derived_thresholds"]}
    assert counts["420"] == sum(metrics["episode_successes"])
    summary = {"status": "complete", "arm": job["arm"], "seed": job["seed"],
               "checkpoint_sha256": job["checkpoint_sha256"], "bn_sha256": job["bn_sha256"],
               "evaluation_recipe_sha256": job["evaluation_recipe_sha256"], "job_sha256": sha(job_path),
               "task_payload_sha256": task_sha, "task_count": len(rows),
               "physical_action_cap": 420, "success_counts_by_threshold": counts,
               "first_hit_records_sha256": sha(out / "first-hit.jsonl"),
               "elapsed_seconds": time.monotonic() - started, "test_access": True,
               "success_definition": recipe["success_event"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
