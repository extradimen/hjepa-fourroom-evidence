"""Continue the frozen P7 M2-to-M2 prefix for exactly 1000 updates."""
import argparse
import copy
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
from src.mechanisms import MacroEffectHead
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.p1_runtime import Runtime
from src.p6_checkpoint import restore, recalibrate_bn, initialize_head
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
    assert job["status"] == "frozen" and job["stage"] == "p7_m2_continuation"
    assert job["arm"] == "M2_to_M2" and job["seed"] in (37, 53, 71, 89)
    assert job["project_id"] == "26bc4d31-99a3-433c-9035-1a5a7fcd5847"
    assert job["steps"] == 1000 and job["test_access"] is False
    protocol = json.loads(bound(job["protocol_path"], job["protocol_sha256"]).read_text())
    assert job["seed"] in protocol["training"]["seeds"]
    arm = next(a for a in protocol["training"]["arms"] if a["id"] == job["arm"])
    assert arm["steps_1_3000"] == "B+M2+M2_branch_data" and arm["steps_3001_4000"] == "B+M2+M2_branch_data"
    authorization = json.loads(bound(job["budget_authorization_path"], job["budget_authorization_sha256"]).read_text())
    assert authorization["status"] == "user_approved" and authorization["cap_seconds"] == job["required_total_cap_seconds"]
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authorization["deadline_utc"])
    recipe = json.loads(bound("configs/p5-factorial-training.json", job["tail_recipe_sha256"]).read_text())
    assert recipe["steps"] == 1000 and recipe["batch_size"] == 32
    for item in job["materials"]:
        bound(item["path"], item["sha256"])
    project = ROOT.parents[1]
    marker = project / "budget-reservations" / f"{job['reservation_id']}.json"
    guard = json.loads(marker.read_text())
    assert guard["status"] == "consumed_before_launch" and guard["pid"] == os.getppid()
    parent = torch.load(bound(job["parent_checkpoint"], job["parent_checkpoint_sha256"]), map_location="cpu", weights_only=False)
    assert parent["step"] == 3000 and parent["seed"] == job["seed"]
    assert parent["arm"] == "M2_to_M2" and parent["parent_checkpoint"] is None
    assert parent["p7_job_sha256"] == job["parent_job_sha256"]
    parent_summary = json.loads(bound(job["parent_summary"], job["parent_summary_sha256"]).read_text())
    assert parent_summary["status"] == "complete" and parent_summary["arm"] == "M2_to_M2"
    assert parent_summary["seed"] == job["seed"] and parent_summary["updates"] == 3000
    assert parent_summary["checkpoint_sha256"] == job["parent_checkpoint_sha256"]
    assert parent_summary["bn_sha256"] == job["parent_bn_sha256"]
    coefficient = parent_summary["M2_coefficient"]
    assert 1e-6 <= coefficient <= 1e6
    buffers = torch.load(bound(job["parent_bn"], job["parent_bn_sha256"]), map_location="cpu", weights_only=False)
    seed = job["seed"]
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
    reference = copy.deepcopy(runtime.model.get_level(1).encoder).requires_grad_(False).eval()
    assert weights_sha(runtime.model) == parent_summary["initial_parameters_sha256"]
    head = initialize_head(lambda: MacroEffectHead(192, 4, 192).cuda(), seed=2011, cuda_devices=(0,))
    runtime.optim.add_param_group({"params": head.parameters(), "lr": float(runtime.cfg.level2.lr),
                                   "name": "effect_head"})
    restore(parent, runtime.model, head, runtime.optim, expected_prefix=(seed, 0, "M2", True))
    runtime.normalizer = parent["normalizer"]
    state = runtime.model.state_dict()
    assert buffers and all(key in state and state[key].shape == value.shape for key, value in buffers.items())
    state.update(buffers)
    runtime.model.load_state_dict(state, strict=True)
    del parent
    initial_sha = weights_sha(runtime.model)
    assert initial_sha == parent_summary["final_parameters_sha256"]
    out = project / "runs" / job["run_name"]
    assert out.parent == project / "runs"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    generator = torch.Generator().manual_seed(seed + 4000)
    loader = DataLoader(runtime.datasets["train"], batch_size=32, shuffle=True,
                        drop_last=True, num_workers=0, generator=generator)
    iterator = iter(loader)
    for step in range(3001, 4001):
        local_step = step - 3000
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        torch.manual_seed(seed * 100000 + step)
        torch.cuda.manual_seed_all(seed * 100000 + step)
        runtime.model.train()
        for module in runtime.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        runtime.optim.zero_grad(set_to_none=True)
        prepared = runtime.batch(batch)
        output = runtime.forward(prepared, "train")
        assert not any(key.startswith("idm_loss") for key in output)
        row = update(runtime, prepared, output, head, reference, coefficient, "M2", seed, step)
        row.update(step=step, local_step=local_step, seconds=time.monotonic() - started)
        with (out / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if local_step == 1 or local_step % 100 == 0:
            print(json.dumps(row), flush=True)
    final_sha = weights_sha(runtime.model)
    checkpoint = {"model": runtime.model.state_dict(), "optimizer": runtime.optim.state_dict(),
                  "head": head.state_dict(), "step": 4000, "seed": seed, "arm": "M2_to_M2", "idm": 0,
                  "normalizer": runtime.normalizer, "config": OmegaConf.to_container(runtime.cfg, resolve=True),
                  "p7_job": job, "p7_job_sha256": sha(job_path),
                  "parent_checkpoint_sha256": job["parent_checkpoint_sha256"],
                  "parent_bn_sha256": job["parent_bn_sha256"],
                  "parent_initial_parameters_sha256": initial_sha,
                  "head_active_from_step": 1, "branch_data_from_step": 1}
    torch.save(checkpoint, out / "latest.pt")
    assert recalibrate_bn(runtime, batches=128) == 128
    assert weights_sha(runtime.model) == final_sha
    bn_buffers = {key: value.cpu().clone() for key, value in runtime.model.state_dict().items()
                  if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    torch.save(bn_buffers, out / "bn-buffers.pt")
    summary = {"status": "complete", "arm": "M2_to_M2", "seed": seed,
               "start_step": 3001, "end_step": 4000, "updates": 1000,
               "parent_checkpoint_sha256": job["parent_checkpoint_sha256"],
               "parent_bn_sha256": job["parent_bn_sha256"],
               "initial_parameters_sha256": initial_sha, "final_parameters_sha256": final_sha,
               "checkpoint_sha256": sha(out / "latest.pt"), "bn_sha256": sha(out / "bn-buffers.pt"),
               "job_sha256": sha(job_path), "M2_coefficient": coefficient,
               "elapsed_seconds": time.monotonic() - started, "bn_recalibration_batches": 128,
               "test_access": False, "closed_loop_evaluated": False,
               "head_active_from_step": 1, "branch_data_from_step": 1}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
