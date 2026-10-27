"""Fresh 3000-step P7 M3-R prefix with a frozen, seed-specific coefficient."""
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

import numpy as np
import torch
from loss import InverseDynamicsModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.p1_runtime import Runtime
from src.p6_checkpoint import initialize_head, recalibrate_bn, snapshot
from src.p6_update import update
from src.upstream_adapter import build_common_baseline


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def bound(path, digest):
    result = (ROOT / path).resolve()
    assert result.is_relative_to(ROOT) and sha(result) == digest
    return result


def weights_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = (ROOT / args.job).resolve()
    assert job_path.is_relative_to(ROOT)
    job = json.loads(job_path.read_text())
    assert job["status"] == "frozen" and job["stage"] == "p7_m3r_prefix"
    assert job["arm"] == "M3R_to_M3R" and job["steps"] == 3000
    assert job["project_id"] == "26bc4d31-99a3-433c-9035-1a5a7fcd5847"
    assert job["test_access"] is False
    protocol = json.loads(bound(job["protocol_path"], job["protocol_sha256"]).read_text())
    seed = job["seed"]
    assert seed in protocol["training"]["seeds"]
    arm = next(a for a in protocol["training"]["arms"] if a["id"] == job["arm"])
    assert arm["steps_1_3000"] == "B+M3-R"
    authorization = json.loads(bound(job["budget_authorization_path"], job["budget_authorization_sha256"]).read_text())
    assert authorization["status"] == "user_approved"
    assert authorization["cap_seconds"] == job["required_total_cap_seconds"]
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authorization["deadline_utc"])
    amendment = json.loads(bound(job["calibration_amendment_path"], job["calibration_amendment_sha256"]).read_text())
    assert amendment["status"] == "frozen_before_auxiliary_training" and seed in amendment["seeds"]
    calibration = json.loads(bound(job["calibration_path"], job["calibration_sha256"]).read_text())
    assert calibration["status"] == "complete" and calibration["world_model_updates"] == 0
    assert calibration["test_access"] is False and calibration["dev_used"] is False
    assert len(calibration["results"]) == 1 and calibration["results"][0]["seed"] == seed
    cal = calibration["results"][0]
    assert cal["state_unchanged"] is True
    assert cal["coefficients"]["M3-R"]["status"] == "ready"
    coefficient = cal["coefficients"]["M3-R"]["coefficient"]
    assert 1e-6 <= coefficient <= 1e6
    recipe = json.loads(bound("configs/p5-common-training.json", job["common_recipe_sha256"]).read_text())
    assert recipe["steps"] == 3000 and recipe["batch_size"] == 32
    for item in job["materials"]:
        bound(item["path"], item["sha256"])
    project = ROOT.parents[1]
    marker = project / "budget-reservations" / f"{job['reservation_id']}.json"
    guard = json.loads(marker.read_text())
    assert guard["status"] == "consumed_before_launch" and guard["pid"] == os.getppid()

    runtime = Runtime(ROOT)
    del runtime.model, runtime.optim
    runtime.cfg.seed = seed
    runtime.settings["common_idm_coefficient"] = 0.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime.model, losses, runtime.cfg = build_common_baseline(runtime.cfg, 0.0, runtime.settings["common_idm_hidden"])
    runtime.model.cuda()
    for key, value in losses.items():
        setattr(runtime, key, value.cuda())
    runtime.optim = torch.optim.AdamW(
        [{"params": runtime.model.get_level(level).parameters(),
          "lr": float(runtime.cfg[f"level{level}"].lr), "name": f"level{level}"}
         for level in (1, 2)], weight_decay=float(runtime.cfg.optimizer.weight_decay))
    initial_sha = weights_sha(runtime.model)
    head = initialize_head(lambda: InverseDynamicsModel(192, 256, 192).cuda(), seed=2011, cuda_devices=(0,))
    runtime.optim.add_param_group({"params": head.parameters(), "lr": float(runtime.cfg.level1.lr),
                                   "name": "predicted_action_head"})
    out = project / "runs" / job["run_name"]
    assert out.parent == project / "runs"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    generator = torch.Generator().manual_seed(seed + 4000)
    loader = DataLoader(runtime.datasets["train"], batch_size=32, shuffle=True,
                        drop_last=True, num_workers=0, generator=generator)
    iterator = iter(loader)
    for step in range(1, 3001):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        torch.manual_seed(seed * 100000 + step)
        torch.cuda.manual_seed_all(seed * 100000 + step)
        runtime.model.train()
        runtime.optim.zero_grad(set_to_none=True)
        prepared = runtime.batch(batch)
        output = runtime.forward(prepared, "train")
        assert not any(key.startswith("idm_loss") for key in output)
        row = update(runtime, prepared, output, head, None, coefficient, "M3-R", seed, step)
        row.update(step=step, seconds=time.monotonic() - started)
        with (out / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if step == 1 or step % 100 == 0:
            print(json.dumps(row), flush=True)
    final_sha = weights_sha(runtime.model)
    payload = snapshot(runtime.model, head, runtime.optim, phase_step=3000,
                       prefix_key=(seed, 0, "M3-R", False))
    payload.update(normalizer=runtime.normalizer,
                   config=OmegaConf.to_container(runtime.cfg, resolve=True),
                   seed=seed, arm="M3R_to_M3R", p7_job=job, p7_job_sha256=sha(job_path),
                   initialization="fresh", initial_parameters_sha256=initial_sha,
                   final_parameters_sha256=final_sha, parent_checkpoint=None)
    torch.save(payload, out / "latest.pt")
    del payload
    assert recalibrate_bn(runtime, batches=128) == 128
    assert weights_sha(runtime.model) == final_sha
    buffers = {key: value.cpu().clone() for key, value in runtime.model.state_dict().items()
               if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    torch.save(buffers, out / "bn-buffers.pt")
    summary = {"status": "complete", "stage": "prefix", "arm": "M3R_to_M3R", "seed": seed,
               "start_step": 1, "end_step": 3000, "updates": 3000,
               "initial_parameters_sha256": initial_sha, "final_parameters_sha256": final_sha,
               "checkpoint_sha256": sha(out / "latest.pt"), "bn_sha256": sha(out / "bn-buffers.pt"),
               "job_sha256": sha(job_path), "calibration_sha256": job["calibration_sha256"],
               "M3R_coefficient": coefficient, "bn_recalibration_batches": 128,
               "elapsed_seconds": time.monotonic() - started, "test_access": False,
               "closed_loop_evaluated": False, "parent_checkpoint": None,
               "head_active_from_step": 1}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
