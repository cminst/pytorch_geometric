from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch_geometric.data import Batch, Data

from umc_pointwavelet import build_model as build_pointwavelet_model
from utils.datasets import build_test_transform_clean
from utils.models import NoWeightClassifier, UMCClassifier


@dataclass
class BenchmarkRow:
    family: str
    model: str
    resolved_device: str
    params: int
    preprocess_flops: float
    forward_flops: float
    total_flops: float
    preprocess_ms: float
    forward_ms: float
    total_ms: float
    relative_flops: float
    relative_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure FLOPs and runtime for minimal spectral baselines and "
            "PointWavelet-L with and without UMC."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to benchmark on: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--spec-dense-points", type=int, default=2048)
    parser.add_argument("--spec-num-points", type=int, default=512)
    parser.add_argument("--spec-knn-k", type=int, default=20)
    parser.add_argument("--spec-K", type=int, default=64)
    parser.add_argument("--pw-num-points", type=int, default=1024)
    parser.add_argument("--umc-hidden", type=str, default="128,32")
    parser.add_argument("--umc-knn", type=int, default=20)
    parser.add_argument("--umc-min-weight", type=float, default=1e-4)
    parser.add_argument("--umc-no-inverse", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    return parser.parse_args()


def parse_int_pair(value: str) -> Tuple[int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated integers, got: {value!r}")
    return int(parts[0]), int(parts[1])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def normalize_points(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    centered = points - points.mean(dim=-2, keepdim=True)
    scale = centered.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(eps)
    return centered / scale.unsqueeze(-1)


def clone_data_list(data_list: Sequence[Data]) -> List[Data]:
    return [data.clone() for data in data_list]


def measure_runtime_ms(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        _ = fn()
    synchronize(device)

    times_ms: List[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        _ = fn()
        synchronize(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return sum(times_ms) / len(times_ms)


def measure_flops(fn: Callable[[], object]) -> float:
    with FlopCounterMode(display=False) as counter:
        _ = fn()
    return float(counter.get_total_flops())


def validate_minimal_spectral_dependencies() -> None:
    missing = []

    try:
        import torch_scatter  # noqa: F401
    except ModuleNotFoundError:
        missing.append("torch_scatter")

    try:
        import torch_cluster  # noqa: F401
    except ModuleNotFoundError:
        missing.append("torch_cluster")

    if missing:
        deps = ", ".join(sorted(missing))
        raise ImportError(
            "Minimal spectral benchmark requires optional dependencies "
            f"that are not installed: {deps}."
        )


def build_minimal_raw_batch(
    *,
    batch_size: int,
    dense_points: int,
    device: torch.device,
) -> List[Data]:
    raw_batch: List[Data] = []
    for _ in range(batch_size):
        pos = torch.randn(dense_points, 3, device=device, dtype=torch.float32)
        raw_batch.append(Data(pos=pos, num_nodes=dense_points))
    return raw_batch


def estimate_minimal_preprocess_flops(
    *,
    batch_size: int,
    num_points: int,
    knn_k: int,
) -> float:
    # FlopCounterMode does not currently account well for the PyG graph-building
    # ops or torch.linalg.eigh in this preprocessing path, so we use an explicit
    # analytical fallback when its measured count is zero.
    n = float(num_points)
    b = float(batch_size)
    k = float(knn_k)

    normalize_scale = b * (20.0 * n)
    pairwise_distance = b * (8.0 * n * n)
    knn_graph = b * (2.0 * k * n)
    laplacian_build = b * (8.0 * n * n)
    symmetric_eigh = b * (9.0 * n * n * n)
    return normalize_scale + pairwise_distance + knn_graph + laplacian_build + symmetric_eigh


def benchmark_minimal_spectral(
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
    dense_points: int,
    num_points: int,
    knn_k: int,
    K: int,
) -> List[BenchmarkRow]:
    validate_minimal_spectral_dependencies()

    transform = build_test_transform_clean(
        num_points=num_points,
        knn_k=knn_k,
        K=K,
        include_phi=True,
        phi_device=device.type,
    )
    raw_batch = build_minimal_raw_batch(
        batch_size=batch_size,
        dense_points=dense_points,
        device=device,
    )

    def preprocess_batch() -> Batch:
        transformed = [transform(data) for data in clone_data_list(raw_batch)]
        return Batch.from_data_list(transformed)

    baseline_batch = preprocess_batch()
    num_classes = 10
    models = [
        ("minimal_spectral_naive", NoWeightClassifier(K=K, num_classes=num_classes)),
        ("minimal_spectral_umc", UMCClassifier(K=K, num_classes=num_classes)),
    ]

    preprocess_flops = measure_flops(preprocess_batch)
    if preprocess_flops <= 0.0:
        preprocess_flops = estimate_minimal_preprocess_flops(
            batch_size=batch_size,
            num_points=num_points,
            knn_k=knn_k,
        )
    preprocess_ms = measure_runtime_ms(
        preprocess_batch,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    rows: List[BenchmarkRow] = []
    baseline_total_flops = None
    baseline_total_ms = None

    for model_name, model in models:
        model = model.to(device)
        model.eval()

        def forward_only() -> object:
            with torch.inference_mode():
                return model(baseline_batch)

        forward_flops = measure_flops(forward_only)
        forward_ms = measure_runtime_ms(
            forward_only,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )

        total_flops = preprocess_flops + forward_flops
        total_ms = preprocess_ms + forward_ms

        if baseline_total_flops is None:
            baseline_total_flops = total_flops
            baseline_total_ms = total_ms

        rows.append(
            BenchmarkRow(
                family="minimal_spectral",
                model=model_name,
                resolved_device=str(device),
                params=count_parameters(model),
                preprocess_flops=preprocess_flops,
                forward_flops=forward_flops,
                total_flops=total_flops,
                preprocess_ms=preprocess_ms,
                forward_ms=forward_ms,
                total_ms=total_ms,
                relative_flops=total_flops / baseline_total_flops,
                relative_ms=total_ms / baseline_total_ms,
            )
        )

    return rows


def build_pointwavelet_input(
    *,
    batch_size: int,
    num_points: int,
    device: torch.device,
) -> torch.Tensor:
    xyz = torch.randn(batch_size, num_points, 3, device=device, dtype=torch.float32)
    return normalize_points(xyz)


def benchmark_pointwavelet(
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
    num_points: int,
    umc_hidden: Tuple[int, int],
    umc_knn: int,
    umc_min_weight: float,
    umc_use_inverse: bool,
) -> List[BenchmarkRow]:
    xyz = build_pointwavelet_input(
        batch_size=batch_size,
        num_points=num_points,
        device=device,
    )

    model_specs = [
        ("pointwavelet_l", False),
        ("pointwavelet_l_umc", True),
    ]

    rows: List[BenchmarkRow] = []
    baseline_flops = None
    baseline_ms = None

    for model_name, use_umc in model_specs:
        model = build_pointwavelet_model(
            use_umc=use_umc,
            wf_learnable=True,
            umc_hidden=umc_hidden,
            umc_knn=umc_knn,
            umc_min_weight=umc_min_weight,
            umc_use_inverse=umc_use_inverse,
            num_classes=10,
        ).to(device)
        model.eval()

        def forward_only() -> object:
            with torch.inference_mode():
                return model(xyz)

        forward_flops = measure_flops(forward_only)
        forward_ms = measure_runtime_ms(
            forward_only,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )

        if baseline_flops is None:
            baseline_flops = forward_flops
            baseline_ms = forward_ms

        rows.append(
            BenchmarkRow(
                family="pointwavelet_l",
                model=model_name,
                resolved_device=str(device),
                params=count_parameters(model),
                preprocess_flops=0.0,
                forward_flops=forward_flops,
                total_flops=forward_flops,
                preprocess_ms=0.0,
                forward_ms=forward_ms,
                total_ms=forward_ms,
                relative_flops=forward_flops / baseline_flops,
                relative_ms=forward_ms / baseline_ms,
            )
        )

    return rows


def run_all_benchmarks(args: argparse.Namespace, device: torch.device) -> List[BenchmarkRow]:
    umc_hidden = parse_int_pair(args.umc_hidden)

    rows = []
    rows.extend(
        benchmark_minimal_spectral(
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            dense_points=args.spec_dense_points,
            num_points=args.spec_num_points,
            knn_k=args.spec_knn_k,
            K=args.spec_K,
        )
    )
    rows.extend(
        benchmark_pointwavelet(
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            num_points=args.pw_num_points,
            umc_hidden=umc_hidden,
            umc_knn=args.umc_knn,
            umc_min_weight=args.umc_min_weight,
            umc_use_inverse=not bool(args.umc_no_inverse),
        )
    )
    return rows


def candidate_devices(requested: str) -> List[torch.device]:
    requested = requested.lower()
    if requested == "auto":
        devices: List[torch.device] = []
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        devices.append(torch.device("cpu"))
        return devices

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("Requested --device mps, but MPS is not available.")

    return [torch.device(requested)]


def run_with_fallback(args: argparse.Namespace) -> List[BenchmarkRow]:
    devices = candidate_devices(args.device)
    last_error: Exception | None = None

    for index, device in enumerate(devices):
        try:
            return run_all_benchmarks(args, device)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            is_last = index == len(devices) - 1
            if args.device != "auto" or is_last:
                raise
            print(
                f"Benchmark failed on {device}: {exc}. Falling back to the next device.",
                file=sys.stderr,
            )

    assert last_error is not None
    raise last_error


def to_serializable_rows(rows: Iterable[BenchmarkRow]) -> List[dict]:
    return [asdict(row) for row in rows]


def format_gflops(value: float) -> str:
    return f"{value / 1e9:.3f}"


def format_params_m(value: int) -> str:
    return f"{value / 1e6:.3f}"


def print_results(rows: Sequence[BenchmarkRow]) -> None:
    headers = [
        "family",
        "model",
        "device",
        "params_m",
        "pre_gflops",
        "fwd_gflops",
        "total_gflops",
        "pre_ms",
        "fwd_ms",
        "total_ms",
        "rel_flops",
        "rel_ms",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.family,
                row.model,
                row.resolved_device,
                format_params_m(row.params),
                format_gflops(row.preprocess_flops),
                format_gflops(row.forward_flops),
                format_gflops(row.total_flops),
                f"{row.preprocess_ms:.3f}",
                f"{row.forward_ms:.3f}",
                f"{row.total_ms:.3f}",
                f"{row.relative_flops:.3f}",
                f"{row.relative_ms:.3f}",
            ]
        )

    widths = []
    for column_idx, header in enumerate(headers):
        column_width = len(header)
        for row in table_rows:
            column_width = max(column_width, len(row[column_idx]))
        widths.append(column_width)

    def render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in table_rows:
        print(render(row))


def write_csv(rows: Sequence[BenchmarkRow], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = to_serializable_rows(rows)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0].keys()))
        writer.writeheader()
        writer.writerows(serializable)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.spec_dense_points <= 0 or args.spec_num_points <= 0 or args.pw_num_points <= 0:
        raise ValueError("Point counts must be positive.")
    if args.spec_knn_k <= 0 or args.spec_K <= 0 or args.umc_knn <= 0:
        raise ValueError("KNN and spectral dimensions must be positive.")

    torch.manual_seed(0)
    rows = run_with_fallback(args)

    print_results(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nSaved CSV to {args.csv}")


if __name__ == "__main__":
    main()
