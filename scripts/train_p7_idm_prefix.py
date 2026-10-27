"""Fresh 3000-step continuous-IDM P7 prefix under the project budget guard."""
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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.p1_runtime import Runtime
from src.p6_checkpoint import recalibrate_bn
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
    assert job["status"] == "frozen" and job["stage"] == "p7_idm_prefix"
    assert job["arm"] == "IDM_to_IDM" and job["steps"] == 3000
    assert job["project_id"] == "26bc4d31-99a3-433c-9035-1a5a7fcd5847"
    assert job["test_access"] is False
    protocol = json.loads(bound(job["protocol_path"], job["protocol_sha256"]).read_text())
    seed = job["seed"]
    assert seed in protocol["training"]["seeds"]
    arm = next(a for a in protocol["training"]["arms"] if a["id"] == job["arm"])
    assert arm["steps_1_3000"] == "B+IDM"
    authorization = json.loads(bound(job["budget_authorization_path"], job["budget_authorization_sha256"]).read_text())
    assert authorization["status"] == "user_approved"
    assert authorization["cap_seconds"] == job["required_total_cap_seconds"]
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authorization["deadline_utc"])
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
    runtime.settings["common_idm_coefficient"] = 1.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime.model, losses, runtime.cfg = build_common_baseline(
        runtime.cfg, 1.0, runtime.settings["common_idm_hidden"])
    runtime.model.cuda()
    for key, value in losses.items():
        setattr(runtime, key, value.cuda())
    runtime.optim = torch.optim.AdamW(
        [{"params": runtime.model.get_level(level).parameters(),
          "lr": float(runtime.cfg[f"level{level}"].lr), "name": f"level{level}"}
         for level in (1, 2)], weight_decay=float(runtime.cfg.optimizer.weight_decay))
    initial_sha = weights_sha(runtime.model)
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
        for module in runtime.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        runtime.optim.zero_grad(set_to_none=True)
        output = runtime.forward(runtime.batch(batch), "train")
        assert any(key.startswith("idm_loss") for key in output)
        loss = output["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        runtime.optim.step()
        row = {"step": step, "seconds": time.monotonic() - started,
               "base_loss": float(loss.detach()), "grad_norm": float(grad_norm)}
        with (out / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if step == 1 or step % 100 == 0:
            print(json.dumps(row), flush=True)
    final_sha = weights_sha(runtime.model)
    checkpoint = {"model": runtime.model.state_dict(), "optimizer": runtime.optim.state_dict(),
                  "step": 3000, "seed": seed, "arm": "IDM_to_IDM", "idm": 1,
                  "normalizer": runtime.normalizer,
                  "config": OmegaConf.to_container(runtime.cfg, resolve=True),
                  "p7_job": job, "p7_job_sha256": sha(job_path),
                  "initialization": "fresh", "initial_parameters_sha256": initial_sha,
                  "final_parameters_sha256": final_sha, "parent_checkpoint": None}
    torch.save(checkpoint, out / "latest.pt")
    del checkpoint
    assert recalibrate_bn(runtime, batches=128) == 128
    assert weights_sha(runtime.model) == final_sha
    buffers = {key: value.cpu().clone() for key, value in runtime.model.state_dict().items()
               if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    torch.save(buffers, out / "bn-buffers.pt")
    summary = {"status": "complete", "stage": "prefix", "arm": "IDM_to_IDM", "seed": seed,
               "start_step": 1, "end_step": 3000, "updates": 3000,
               "initial_parameters_sha256": initial_sha, "final_parameters_sha256": final_sha,
               "checkpoint_sha256": sha(out / "latest.pt"),
               "bn_sha256": sha(out / "bn-buffers.pt"), "job_sha256": sha(job_path),
               "idm_coefficient": 1.0, "bn_recalibration_batches": 128,
               "elapsed_seconds": time.monotonic() - started, "test_access": False,
               "closed_loop_evaluated": False, "parent_checkpoint": None,
               "idm_active_from_step": 1}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
