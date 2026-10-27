"""Fresh P7 B-to-B 1000-step tail; invoke only via project guard and global controller."""
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
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.p1_runtime import Runtime
from src.upstream_adapter import build_common_baseline


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def bound(path, digest):
    resolved = (ROOT / path).resolve()
    assert resolved.is_relative_to(ROOT) and sha(resolved) == digest
    return resolved


def weights_sha(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = (ROOT / args.job).resolve()
    assert job_path.is_relative_to(ROOT)
    job = json.loads(job_path.read_text())
    assert job["status"] == "frozen" and job["stage"] == "p7_b_tail"
    assert job["arm"] == "B_to_B" and job["test_access"] is False
    protocol = json.loads(bound(job["protocol_path"], job["protocol_sha256"]).read_text())
    assert job["seed"] in protocol["training"]["seeds"]
    authorization = json.loads(bound(job["budget_authorization_path"], job["budget_authorization_sha256"]).read_text())
    assert authorization["status"] == "user_approved" and authorization["cap_seconds"] == job["required_total_cap_seconds"]
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(authorization["deadline_utc"])
    recipe = json.loads(bound("configs/p5-factorial-training.json", job["tail_recipe_sha256"]).read_text())
    assert recipe["steps"] == 1000 and recipe["batch_size"] == 32
    project = ROOT.parents[1]
    marker = project / "budget-reservations" / f"{job['reservation_id']}.json"
    guard = json.loads(marker.read_text())
    assert guard["status"] == "consumed_before_launch" and guard["pid"] == os.getppid()
    parent_path = bound(job["parent_checkpoint"], job["parent_checkpoint_sha256"])
    bn_path = bound(job["parent_bn"], job["parent_bn_sha256"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    seed = job["seed"]
    assert parent["step"] == 3000 and parent["seed"] == seed
    assert parent["parent_checkpoint"] is None and parent["p7_job"]["arm"] == "B"
    assert parent["settings"]["common_idm_coefficient"] == 0.0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime = Runtime(ROOT)
    del runtime.model, runtime.optim
    runtime.cfg.seed = seed
    runtime.model, losses, runtime.cfg = build_common_baseline(runtime.cfg, 0.0, runtime.settings["common_idm_hidden"])
    runtime.model.cuda()
    for key, value in losses.items():
        setattr(runtime, key, value.cuda())
    runtime.optim = torch.optim.AdamW(
        [{"params": runtime.model.get_level(level).parameters(),
          "lr": float(runtime.cfg[f"level{level}"].lr), "name": f"level{level}"}
         for level in (1, 2)],
        weight_decay=float(runtime.cfg.optimizer.weight_decay),
    )
    runtime.model.load_state_dict(parent["model"], strict=True)
    runtime.optim.load_state_dict(parent["optimizer"])
    runtime.normalizer = parent["normalizer"]
    state = runtime.model.state_dict()
    buffers = torch.load(bn_path, map_location="cpu", weights_only=False)
    assert buffers and all(key in state and state[key].shape == value.shape for key, value in buffers.items())
    state.update(buffers)
    runtime.model.load_state_dict(state, strict=True)
    del parent
    parent_weights = weights_sha(runtime.model)
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
        torch.manual_seed(seed * 100000 + local_step)
        torch.cuda.manual_seed_all(seed * 100000 + local_step)
        runtime.model.train()
        for module in runtime.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        runtime.optim.zero_grad(set_to_none=True)
        output = runtime.forward(runtime.batch(batch), "train")
        assert not any(key.startswith("idm_loss") for key in output)
        loss = output["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(), 1.0, error_if_nonfinite=True)
        runtime.optim.step()
        row = {"step": step, "local_step": local_step, "seconds": time.monotonic() - started,
               "base_loss": float(loss.detach()), "grad_norm": float(grad_norm)}
        with (out / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if local_step == 1 or local_step % 100 == 0:
            print(json.dumps(row), flush=True)
    final_weights = weights_sha(runtime.model)
    checkpoint = {"model": runtime.model.state_dict(), "optimizer": runtime.optim.state_dict(),
                  "step": 4000, "seed": seed, "arm": "B_to_B", "idm": 0,
                  "normalizer": runtime.normalizer, "config": OmegaConf.to_container(runtime.cfg, resolve=True),
                  "p7_job": job, "p7_job_sha256": sha(job_path),
                  "parent_checkpoint_sha256": job["parent_checkpoint_sha256"],
                  "parent_bn_sha256": job["parent_bn_sha256"], "parent_initial_parameters_sha256": parent_weights}
    torch.save(checkpoint, out / "latest.pt")
    runtime.model.requires_grad_(False)
    for module in runtime.model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None
            module.train()
    with torch.no_grad():
        for index, batch in enumerate(runtime.loader("train", seed=50131)):
            if index >= 128:
                break
            torch.manual_seed(11013 + index)
            torch.cuda.manual_seed_all(11013 + index)
            runtime.forward(runtime.batch(batch), "bn_recalibration")
    runtime.model.eval()
    assert weights_sha(runtime.model) == final_weights
    bn_buffers = {key: value.cpu().clone() for key, value in runtime.model.state_dict().items()
                  if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    torch.save(bn_buffers, out / "bn-buffers.pt")
    summary = {"status": "complete", "arm": "B_to_B", "seed": seed, "start_step": 3001,
               "end_step": 4000, "updates": 1000, "parent_checkpoint_sha256": job["parent_checkpoint_sha256"],
               "parent_bn_sha256": job["parent_bn_sha256"], "initial_parameters_sha256": parent_weights,
               "final_parameters_sha256": final_weights, "checkpoint_sha256": sha(out / "latest.pt"),
               "bn_sha256": sha(out / "bn-buffers.pt"), "job_sha256": sha(job_path),
               "elapsed_seconds": time.monotonic() - started, "bn_recalibration_batches": 128,
               "test_access": False, "closed_loop_evaluated": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
