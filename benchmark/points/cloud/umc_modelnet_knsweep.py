import os
import random
import resource
import subprocess
import time

import requests
from burla import remote_parallel_map
from rich import print as rich_print
from rich.panel import Panel


def increase_openfiles_limit(new_value: int):
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_value, hard_limit))


def setup_repo(
    repo_url: str,
    repo_name: str,
    branch_name: str,
    commit_sha: str | None
):
    time.sleep(random.randint(2, 20))  # shorter jitter is enough here

    os.chdir("/workspace")
    if os.path.exists(repo_name):
        print(f"Repository '{repo_name}' already exists")
    else:
        print(f"Cloning repository '{repo_name}'...")
        subprocess.run(["git", "clone", repo_url], check=True)
        os.chdir(repo_name)
        subprocess.run(["git", "checkout", branch_name], check=True)
        os.chdir("..")

    os.chdir(repo_name)
    if commit_sha:
        subprocess.run(["git", "checkout", commit_sha], check=True)


def upload_file_to_centralfile(file_path, destination_name, project_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        content = file.read()

    api_url = "https://qingy1337--centralfile-flask-app.modal.run/upload"
    payload = {
        "project_name": project_name,
        "file_name": destination_name,
        "content": content,
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        print(f"Uploaded '{file_path}' -> '{project_name}/{destination_name}'")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def train_umc(cfg):
    increase_openfiles_limit(100_000)

    setup_repo(
        repo_url=cfg["repo_url"],
        repo_name=cfg["repo_name"],
        branch_name=cfg["branch_name"],
        commit_sha=cfg["commit_sha"],
    )
    os.chdir("benchmark/points")

    methods_slug = cfg["methods"].replace(",", "+")
    seeds_slug = cfg["seeds"].replace(",", "-")

    results_csv = (
        f"umc_kn_sweep_{cfg['dataset']}_{cfg['sweep_axis']}"
        f"_K{cfg['K']}_N{cfg['num_points']}"
        f"_train-{cfg['train_mode']}"
        f"_lambda{cfg['lambda_ortho_grid']}"
        f"_methods-{methods_slug}"
        f"_seeds-{seeds_slug}.csv"
    )

    cmd = [
        "python3", "run_all_umc_experiments.py",
        "--dataset", cfg["dataset"],
        "--train_mode", cfg["train_mode"],
        "--max_bias_train", str(cfg["max_bias_train"]),
        "--dense_points", str(cfg["dense_points"]),
        "--num_points", str(cfg["num_points"]),
        "--K", str(cfg["K"]),
        "--lambda_ortho_grid", str(cfg["lambda_ortho_grid"]),
        "--methods", cfg["methods"],
        "--seeds", cfg["seeds"],
        "--bias_levels", cfg["bias_levels"],
        "--results_csv", results_csv,
        "-v",
    ]

    print("[Burla] Running config:")
    print(cfg)
    subprocess.run(cmd, check=True)

    upload_file_to_centralfile(
        file_path=results_csv,
        destination_name=results_csv,
        project_name="modelnet10_umc_kn_sweeps",
    )


def get_data():
    base = dict(
        repo_url=f"https://{os.environ['GITHUB_PAT']}@github.com/cminst/pytorch_geometric.git",
        repo_name="pytorch_geometric",
        branch_name="main",
        commit_sha=None,

        # Protocol A-ish setup
        dataset="ModelNet10",
        train_mode="aug",
        max_bias_train=3.0,
        dense_points=2048,

        # keep these fixed across the sweep
        lambda_ortho_grid=0,   # replace with your paper's chosen value if not 0
        methods="cap,umc",
        seeds="41,42,43",
        bias_levels="0,1,2,3,4",
    )

    runs = []

    # 1) K sweep at fixed N=512
    for K in [32, 64, 128]:
        runs.append({
            **base,
            "sweep_axis": "K",
            "K": K,
            "num_points": 512,
        })

    # 2) N sweep at fixed K=64
    for N in [256, 512, 1024]:
        runs.append({
            **base,
            "sweep_axis": "N",
            "K": 64,
            "num_points": N,
        })

    rich_print(f"[bold yellow]Testing {len(runs)} configurations...[/bold yellow]")
    return runs


outputs = remote_parallel_map(
    train_umc,
    get_data(),
    func_cpu=32,
    detach=True,
)

rich_print(Panel.fit("[bold green]All configs launched successfully![/bold green]"))
rich_print(f"[bold cyan]Number of configs launched: {len(outputs)}[/bold cyan]\n")
