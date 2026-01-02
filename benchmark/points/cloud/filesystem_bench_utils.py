import os
import json
import time
import shutil
import zipfile
import argparse
from pathlib import Path
from dataclasses import dataclass

from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


@dataclass
class BenchmarkResult:
    """Stores timing results for each operation"""
    operation: str
    duration: float
    details: str = ""
    files_processed: int = 0
    bytes_processed: int = 0

    @property
    def throughput_files(self) -> float:
        if self.duration > 0 and self.files_processed > 0:
            return self.files_processed / self.duration
        return 0.0

    @property
    def throughput_mb(self) -> float:
        if self.duration > 0 and self.bytes_processed > 0:
            return (self.bytes_processed / (1024 * 1024)) / self.duration
        return 0.0


class FileOperationsBenchmark:
    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir).resolve()
        self.results: list[BenchmarkResult] = []
        self.dataset = None

    def ensure_dir(self):
        """Ensure benchmark directory exists"""
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self):
        """Download the dataset (not timed)"""
        console.print("[bold blue]Downloading dataset (not timed)...[/bold blue]")
        cache_dir = self.benchmark_dir / "hf_cache"
        self.dataset = load_dataset(
            "yahma/alpaca-cleaned",
            cache_dir=str(cache_dir)
        )
        console.print("[green]Dataset downloaded and cached![/green]\n")

    def benchmark_load_dataset(self) -> BenchmarkResult:
        """Benchmark loading the dataset from cache"""
        cache_dir = self.benchmark_dir / "hf_cache"

        start = time.perf_counter()

        # Load from cache
        dataset = load_dataset(
            "yahma/alpaca-cleaned",
            cache_dir=str(cache_dir)
        )

        # Access all data to ensure it's fully loaded
        total_records = 0
        total_bytes = 0
        for split in dataset:
            for record in dataset[split]:
                total_records += 1
                # Estimate bytes from record content
                total_bytes += len(json.dumps(record).encode())

        duration = time.perf_counter() - start

        return BenchmarkResult(
            operation="Load Dataset",
            duration=duration,
            details=f"yahma/alpaca-cleaned ({total_records:,} records)",
            files_processed=total_records,
            bytes_processed=total_bytes
        )

    def benchmark_write_jsonl_files(self, num_files: int = 100, file_size_mb: int = 5) -> BenchmarkResult:
        """Write placeholder JSONL files"""
        jsonl_dir = self.benchmark_dir / "jsonl_files"
        jsonl_dir.mkdir(exist_ok=True)

        # Create sample record template
        sample_record = {
            "id": 0,
            "instruction": "This is a placeholder instruction " * 20,
            "input": "Sample input text " * 30,
            "output": "Sample output response " * 50,
            "metadata": {"source": "benchmark", "version": "1.0"}
        }
        sample_line = json.dumps(sample_record) + "\n"
        line_size = len(sample_line.encode())

        target_size = file_size_mb * 1024 * 1024
        lines_per_file = target_size // line_size

        start = time.perf_counter()

        total_bytes = 0
        for i in range(num_files):
            file_path = jsonl_dir / f"data_{i:04d}.jsonl"
            with open(file_path, 'w') as f:
                for j in range(lines_per_file):
                    record = sample_record.copy()
                    record["id"] = i * lines_per_file + j
                    line = json.dumps(record) + "\n"
                    f.write(line)
                    total_bytes += len(line.encode())

        duration = time.perf_counter() - start

        return BenchmarkResult(
            operation="Write JSONL Files",
            duration=duration,
            details=f"{num_files} files x {file_size_mb}MB each",
            files_processed=num_files,
            bytes_processed=total_bytes
        )

    def benchmark_zip_files(self) -> BenchmarkResult:
        """Zip all JSONL files"""
        jsonl_dir = self.benchmark_dir / "jsonl_files"
        zip_path = self.benchmark_dir / "jsonl_archive.zip"

        files = list(jsonl_dir.glob("*.jsonl"))
        total_bytes_input = sum(f.stat().st_size for f in files)

        start = time.perf_counter()

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file_path in files:
                zf.write(file_path, file_path.name)

        duration = time.perf_counter() - start

        zip_size = zip_path.stat().st_size
        compression_ratio = (1 - zip_size / total_bytes_input) * 100 if total_bytes_input > 0 else 0

        return BenchmarkResult(
            operation="Zip Files",
            duration=duration,
            details=f"{len(files)} files -> {zip_size / (1024**2):.1f}MB ({compression_ratio:.1f}% compression)",
            files_processed=len(files),
            bytes_processed=total_bytes_input
        )

    def benchmark_unzip_files(self) -> BenchmarkResult:
        """Unzip the archive"""
        zip_path = self.benchmark_dir / "jsonl_archive.zip"
        extract_dir = self.benchmark_dir / "jsonl_extracted"
        extract_dir.mkdir(exist_ok=True)

        zip_size = zip_path.stat().st_size

        start = time.perf_counter()

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        duration = time.perf_counter() - start

        files = list(extract_dir.glob("*.jsonl"))
        total_bytes = sum(f.stat().st_size for f in files)

        return BenchmarkResult(
            operation="Unzip Files",
            duration=duration,
            details=f"{zip_size / (1024**2):.1f}MB -> {len(files)} files ({total_bytes / (1024**2):.1f}MB)",
            files_processed=len(files),
            bytes_processed=total_bytes
        )

    def benchmark_copy_files(self, num_copies: int = 5) -> BenchmarkResult:
        """Create multiple copies of all files"""
        jsonl_dir = self.benchmark_dir / "jsonl_files"

        # Get source stats
        source_files = list(jsonl_dir.glob("*.jsonl"))
        source_bytes = sum(f.stat().st_size for f in source_files)

        start = time.perf_counter()

        total_files = 0
        total_bytes = 0

        for i in range(num_copies):
            copy_dir = self.benchmark_dir / f"copy_{i}"
            shutil.copytree(jsonl_dir, copy_dir)
            total_files += len(source_files)
            total_bytes += source_bytes

        duration = time.perf_counter() - start

        return BenchmarkResult(
            operation="Copy Files (x5)",
            duration=duration,
            details=f"{num_copies} complete copies created",
            files_processed=total_files,
            bytes_processed=total_bytes
        )

    def benchmark_delete_all(self) -> BenchmarkResult:
        """Delete everything in the benchmark directory"""
        # Count files and bytes before deletion
        total_files = 0
        total_bytes = 0

        for root, dirs, files in os.walk(self.benchmark_dir):
            for f in files:
                file_path = Path(root) / f
                try:
                    total_bytes += file_path.stat().st_size
                    total_files += 1
                except OSError:
                    pass

        start = time.perf_counter()

        # Delete everything in the directory
        for item in self.benchmark_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        duration = time.perf_counter() - start

        return BenchmarkResult(
            operation="Delete All",
            duration=duration,
            details=f"Removed {total_files:,} files ({total_bytes / (1024**2):.1f}MB)",
            files_processed=total_files,
            bytes_processed=total_bytes
        )

    def run_benchmarks(self):
        """Run all benchmarks"""
        self.ensure_dir()

        console.print(Panel.fit(
            f"[bold cyan]File Operations Benchmark[/bold cyan]\n"
            f"Directory: [yellow]{self.benchmark_dir}[/yellow]",
            title="Starting Benchmark",
            border_style="blue"
        ))
        console.print()

        # Download dataset (not timed)
        self.download_dataset()

        console.print("[bold yellow]Starting timed benchmarks...[/bold yellow]\n")

        total_start = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Running benchmarks...", total=6)

            progress.update(task, description="Loading dataset from cache...")
            self.results.append(self.benchmark_load_dataset())
            progress.advance(task)

            progress.update(task, description="Writing 100 JSONL files (5MB each)...")
            self.results.append(self.benchmark_write_jsonl_files(num_files=100, file_size_mb=5))
            progress.advance(task)

            progress.update(task, description="Zipping files...")
            self.results.append(self.benchmark_zip_files())
            progress.advance(task)

            progress.update(task, description="Unzipping files...")
            self.results.append(self.benchmark_unzip_files())
            progress.advance(task)

            progress.update(task, description="Creating 5 copies...")
            self.results.append(self.benchmark_copy_files(num_copies=5))
            progress.advance(task)

            progress.update(task, description="Deleting all files...")
            self.results.append(self.benchmark_delete_all())
            progress.advance(task)

        total_duration = time.perf_counter() - total_start

        self.print_results(total_duration)

    def print_results(self, total_duration: float):
        """Print detailed results table"""
        console.print("\n")

        # Main results table
        table = Table(
            title="Detailed Benchmark Results",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        table.add_column("Operation", style="cyan", width=22)
        table.add_column("Duration", justify="right", style="green", width=12)
        table.add_column("% of Total", justify="right", style="yellow", width=10)
        table.add_column("Details", style="dim", width=45)
        table.add_column("Files", justify="right", width=10)
        table.add_column("Files/sec", justify="right", style="blue", width=12)
        table.add_column("MB/sec", justify="right", style="magenta", width=10)

        for result in self.results:
            pct = (result.duration / total_duration) * 100 if total_duration > 0 else 0
            table.add_row(
                result.operation,
                f"{result.duration:.3f}s",
                f"{pct:.1f}%",
                result.details,
                f"{result.files_processed:,}" if result.files_processed else "-",
                f"{result.throughput_files:,.1f}" if result.throughput_files else "-",
                f"{result.throughput_mb:.1f}" if result.throughput_mb else "-"
            )

        console.print(table)

        # Summary statistics
        total_files = sum(r.files_processed for r in self.results)
        total_bytes = sum(r.bytes_processed for r in self.results)

        summary = Table(
            title="Summary Statistics",
            show_header=True,
            header_style="bold green",
            border_style="green"
        )
        summary.add_column("Metric", style="cyan", width=30)
        summary.add_column("Value", justify="right", style="yellow", width=25)

        summary.add_row("Total Benchmark Duration", f"{total_duration:.3f} seconds")
        summary.add_row("Total Files Processed", f"{total_files:,}")
        summary.add_row("Total Data Processed", f"{total_bytes / (1024**2):,.1f} MB")
        summary.add_row("Total Data Processed", f"{total_bytes / (1024**3):,.2f} GB")
        summary.add_row("Avg Throughput (Files)", f"{total_files / total_duration:,.1f} files/sec")
        summary.add_row("Avg Throughput (Data)", f"{(total_bytes / (1024**2)) / total_duration:,.1f} MB/sec")

        console.print("\n")
        console.print(summary)

        # Performance analysis
        analysis = Table(
            title="Performance Analysis",
            show_header=True,
            header_style="bold red",
            border_style="red"
        )
        analysis.add_column("Analysis", style="cyan", width=25)
        analysis.add_column("Result", style="yellow", width=50)

        # Find slowest and fastest
        slowest = max(self.results, key=lambda r: r.duration)
        fastest = min(self.results, key=lambda r: r.duration)

        # Find best throughput
        best_file_throughput = max(self.results, key=lambda r: r.throughput_files)
        best_mb_throughput = max(self.results, key=lambda r: r.throughput_mb)

        analysis.add_row(
            "Slowest Operation",
            f"{slowest.operation} ({slowest.duration:.3f}s)"
        )
        analysis.add_row(
            "Fastest Operation",
            f"{fastest.operation} ({fastest.duration:.3f}s)"
        )
        analysis.add_row(
            "Best File Throughput",
            f"{best_file_throughput.operation} ({best_file_throughput.throughput_files:,.1f} files/sec)"
        )
        analysis.add_row(
            "Best Data Throughput",
            f"{best_mb_throughput.operation} ({best_mb_throughput.throughput_mb:,.1f} MB/sec)"
        )

        # I/O bound analysis
        io_ops = ["Write JSONL Files", "Zip Files", "Unzip Files", "Copy Files (x5)", "Delete All"]
        io_time = sum(r.duration for r in self.results if r.operation in io_ops)
        io_pct = (io_time / total_duration) * 100 if total_duration > 0 else 0

        analysis.add_row(
            "I/O Operations Time",
            f"{io_time:.3f}s ({io_pct:.1f}% of total)"
        )

        console.print("\n")
        console.print(analysis)

        console.print("\n[bold green]Benchmark completed successfully![/bold green]\n")


def benchmark_filesystem(base_dir: str = './benchmark_data'):
    try:
        benchmark = FileOperationsBenchmark(base_dir)
        benchmark.run_benchmarks()
    except KeyboardInterrupt:
        console.print("\n[red]Benchmark interrupted by user.[/red]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise
