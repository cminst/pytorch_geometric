import os
import random
import resource
import shutil
import subprocess
import time

import requests
from burla import remote_parallel_map
from rich import print as rich_print
from rich.panel import Panel


def increase_openfiles_limit(new_value: int):
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_value, hard_limit))  # Increase soft limit


def run_benchmark(inputs):
    subprocess.run(
        ['wget', 'https://raw.githubusercontent.com/qingy1337/shellscript/refs/heads/main/filesystem_bench_utils.py'],
        check=True
    )
    from filesystem_bench_utils import benchmark_filesystem
    increase_openfiles_limit(100000)
    if not os.path.exists('/workspace/shared/filesys_bench'):
        os.mkdir('/workspace/shared/filesys_bench')
    benchmark_filesystem('/workspace/shared/filesys_bench')

# ----------------------------------------------------------

outputs = remote_parallel_map(run_benchmark, [('test')], func_cpu=2)

rich_print(Panel.fit("[bold green]All runs processed successfully![/bold green]"))
rich_print(f"[bold cyan]Number of configs tested: {len(outputs)}[/bold cyan]\n")
