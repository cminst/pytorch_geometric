"""Quick CPU vs GPU benchmark for torch.linalg.eigh on symmetric NxN matrices."""

import argparse
import time
from typing import Iterable, List

import torch
from rich.console import Console
from rich.table import Table


def parse_ns(ns_str: str) -> List[int]:
    return [int(x) for x in ns_str.split(",") if x.strip()]


def time_eigh(n: int, device: torch.device, runs: int) -> tuple[float, float]:
    """Returns (mean_sec, best_sec) over runs."""
    # single warmup to prime kernels and allocator
    A = torch.randn(n, n, device=device)
    M = (A + A.T) * 0.5
    with torch.no_grad():
        torch.linalg.eigh(M)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(runs):
        A = torch.randn(n, n, device=device)
        M = (A + A.T) * 0.5
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            torch.linalg.eigh(M)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / max(len(times), 1)
    best = min(times) if times else 0.0
    return mean, best


def main(ns: Iterable[int], runs: int, do_cpu: bool, do_gpu: bool) -> None:
    devices = []
    if do_cpu:
        devices.append(torch.device("cpu"))
    if do_gpu and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if do_gpu and not torch.cuda.is_available():
        print("GPU requested but CUDA is not available; skipping GPU.")

    console = Console()
    table = Table(title="torch.linalg.eigh benchmark")
    table.add_column("N", justify="right")
    table.add_column("device")
    table.add_column("mean ms", justify="right")
    table.add_column("best ms", justify="right")

    for n in ns:
        for dev in devices:
            mean, best = time_eigh(n, dev, runs=runs)
            table.add_row(
                str(n),
                dev.type,
                f"{mean*1000:.2f}",
                f"{best*1000:.2f}",
            )

    console.print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark torch.linalg.eigh on CPU vs GPU.")
    ap.add_argument("--ns", type=str, default="128,256,512,1024", help="Comma-separated sizes N to test.")
    ap.add_argument("--runs", type=int, default=5, help="Timed runs per (N, device).")
    ap.add_argument("--cpu", action="store_true", help="Include CPU.")
    ap.add_argument("--gpu", action="store_true", help="Include GPU (if available).")
    args = ap.parse_args()

    # Default: both if none explicitly selected.
    do_cpu = args.cpu or (not args.cpu and not args.gpu)
    do_gpu = args.gpu or (not args.cpu and not args.gpu)

    main(parse_ns(args.ns), runs=args.runs, do_cpu=do_cpu, do_gpu=do_gpu)
