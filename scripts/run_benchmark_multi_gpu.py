"""Run the benchmark with one worker process per GPU (e.g. a Kaggle notebook with 2x T4).

The scenes are dealt out to the workers; every worker writes its runs under the same output
directory and the tables are collected at the end. Runs are resumable: a (scene, variant)
pair that already has a ``metrics.json`` is skipped, so the script can simply be restarted.
Without a GPU, ``--workers`` CPU processes share the cores.

Examples::

    python scripts/run_benchmark_multi_gpu.py --out results/gpu --profile gpu --lpips alex \\
        depth=learned depth_checkpoint=assets/checkpoints/tiny_depth.pth
    python scripts/run_benchmark_multi_gpu.py --out results/cpu/seed1 --profile cpu \\
        --gpus 0 --workers 4 --seed 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from recon4d.benchmark import DEFAULT_VARIANTS, collect_results
from recon4d.data.synthetic import SCENE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/gpu")
    parser.add_argument("--profile", default="gpu")
    parser.add_argument("--scenes", default=",".join(SCENE_NAMES))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--lpips", default=None, help="LPIPS backbone (alex, vgg, squeeze)")
    parser.add_argument("--gpus", type=int, default=None, help="number of GPUs to use")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes, without a GPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*", help="configuration overrides, key=value")
    args = parser.parse_args()

    n_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    scenes = args.scenes.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = n_gpus if n_gpus > 0 else max(args.workers, 1)
    processes = []
    for worker in range(workers):
        mine = scenes[worker::workers]
        if not mine:
            continue
        command = [
            sys.executable,
            "-m",
            "recon4d.cli",
            "benchmark",
            "--out",
            str(out),
            "--profile",
            args.profile,
            "--scenes",
            ",".join(mine),
            "--variants",
            args.variants,
            "--device",
            "cuda" if n_gpus > 0 else "cpu",
            "--seed",
            str(args.seed),
        ]
        if args.lpips:
            command += ["--lpips", args.lpips]
        command += args.overrides
        env = dict(os.environ)
        if n_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(worker)
        else:  # the CPU workers share the cores instead of each claiming all of them
            env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // workers))
        log = (out / f"worker{worker}.log").open("w")
        print(f"worker {worker}: scenes {mine} -> {log.name}", flush=True)
        processes.append(
            (subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )

    failed = False
    for process, log in processes:
        failed |= process.wait() != 0
        log.close()
    results = collect_results(out, args.profile)
    print(f"collected {sum(len(v) for v in results.values())} runs in {out / 'results.md'}")
    if failed:
        sys.exit("at least one worker failed: see the worker*.log files")


if __name__ == "__main__":
    main()
