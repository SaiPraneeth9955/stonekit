"""Measure GPU memory and time for candidate 3D U-Nets on this machine. Reads no scan.

Inputs are random tensors, for timing only. A GPU-only number is a lower bound on real training
time, because it leaves out data loading and augmentation.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from stonekit.config import create_run_folder, load_yaml


def strides_for(stages: int) -> list[list[int]]:
    """In-plane only for the first two poolings, because slices are thicker than pixels."""
    strides = [[1, 1, 1]]
    for stage in range(1, stages):
        strides.append([1, 2, 2] if stage <= 2 else [2, 2, 2])
    return strides


def kernels_for(stages: int) -> list[list[int]]:
    return [[1, 3, 3] if stage <= 1 else [3, 3, 3] for stage in range(stages)]


def build(candidate: dict) -> nn.Module:
    stages = candidate["stages"]
    return PlainConvUNet(
        input_channels=1,
        n_stages=stages,
        features_per_stage=[
            min(candidate["base_features"] * 2**s, candidate["max_features"]) for s in range(stages)
        ],
        conv_op=nn.Conv3d,
        kernel_sizes=kernels_for(stages),
        strides=strides_for(stages),
        n_conv_per_stage=[2] * stages,
        num_classes=2,
        n_conv_per_stage_decoder=[2] * (stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def measure(candidate: dict, config: dict, device: torch.device) -> dict:
    result = {**candidate}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build(candidate).to(device)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.SGD(
            model.parameters(), lr=config["learning_rate"], momentum=0.99, nesterov=True
        )
        scaler = torch.amp.GradScaler("cuda")
        loss_fn = nn.CrossEntropyLoss()
        shape = (candidate["batch"], 1, *candidate["patch"])
        x = torch.randn(shape, device=device)
        y = torch.randint(0, 2, (candidate["batch"], *candidate["patch"]), device=device)
        times = []
        total = config["warmup_iterations"] + config["measured_iterations"]
        for iteration in range(total):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            torch.cuda.synchronize()
            if iteration >= config["warmup_iterations"]:
                times.append(time.perf_counter() - start)
        result["seconds_per_iteration_gpu_only"] = sum(times) / len(times)
        result["train_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["train_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20

        del optimiser, scaler, x, y, loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model(torch.randn((1, 1, *candidate["patch"]), device=device))
        result["inference_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        result["outcome"] = "ran"
        del model
    except torch.cuda.OutOfMemoryError:
        result["outcome"] = "out_of_memory"
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")
    results = [measure(candidate, config, device) for candidate in config["candidates"]]
    summary = {
        "note": "Random tensors, timing only. GPU-only time excludes data loading and augmentation.",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "python": platform.python_version(),
        "results": results,
    }
    folder = create_run_folder(Path(config["runs_folder"]), config["id"])
    (folder / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (folder / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved to", folder)


if __name__ == "__main__":
    main()
