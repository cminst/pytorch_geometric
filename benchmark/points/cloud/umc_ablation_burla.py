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
    commit_sha: str | None,
):
    """Sets up the repository."""
    time.sleep(random.randint(2, 120))

    os.chdir("/workspace")
    if os.path.exists(repo_name):
        print(f"Repository '{repo_name}' already exists")
    else:
        print(f"Cloning repository '{repo_name}'...")
        subprocess.run(["git", "clone", repo_url])

        os.chdir(repo_name)
        subprocess.run(["git", "checkout", branch_name])
        os.chdir("..")

    os.chdir(repo_name)
    if commit_sha:
        subprocess.run(["git", "checkout", commit_sha])


def upload_file_to_centralfile(file_path, destination_name, project_name):
    """Uploads a file to a project directory in Centralfile."""
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
        print(f"Success! File '{file_path}' uploaded to '{project_name}/{destination_name}'")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def train_umc(inputs_dict):
    """Runs a UMC ablation grid from the given config `inputs_dict`."""
    increase_openfiles_limit(100_000)

    setup_repo(
        repo_url=inputs_dict["repo_url"],
        repo_name=inputs_dict["repo_name"],
        branch_name=inputs_dict["branch_name"],
        commit_sha=inputs_dict["commit_sha"],
    )
    os.chdir("benchmark/points")

    print("[Burla] Running with config ---")
    print(inputs_dict)
    print("-------------------------------")

    results_csv = inputs_dict["results_csv"]
    cmd = [
        "python3",
        "run_umc_ablation_grid.py",
        "--dataset",
        inputs_dict["dataset"],
        "--train_mode",
        inputs_dict["train_mode"],
        "--lambda_ortho_grid",
        str(inputs_dict["lambda_ortho_grid"]),
        "--umc_configs",
        inputs_dict["umc_config"],
        "--degree_features",
        inputs_dict["degree_features"],
        "--seeds",
        str(inputs_dict["seeds"]),
        "--results_csv",
        results_csv,
    ]
    subprocess.run(cmd)

    upload_file_to_centralfile(
        file_path=results_csv,
        destination_name=inputs_dict["centralfile_name"],
        project_name=inputs_dict["centralfile_project"],
    )


def get_data():
    from burla_io import prepare_inputs

    params_to_test = dict(
        repo_url=[f"https://{os.environ['GITHUB_PAT']}@github.com/cminst/pytorch_geometric.git"],
        repo_name=["pytorch_geometric"],
        branch_name=["umc_ft_ablation"],
        commit_sha=[None],
        # ------------------------- Script parameters
        dataset=["ModelNet10", "ModelNet40", "ScanObjectNN"],
        train_mode=["clean"],
        lambda_ortho_grid=["0"],
        umc_config=["full_point_encoder"],
        degree_features=["log"],
        seeds=list(range(10)),
        centralfile_project=["umc_ablation_burla"],
    )

    sweep_runs = prepare_inputs(params_to_test)

    filtered_runs = []
    for run in sweep_runs:
        degree_features = run["degree_features"]
        if run["umc_config"] != "deg_only" and degree_features != "log":
            continue

        run["results_csv"] = (
            f"umc_ablation_{run['dataset']}_{run['train_mode']}_{run['umc_config']}_"
            f"{degree_features}_seed{run['seeds']}.csv"
        )
        run["centralfile_name"] = run["results_csv"]
        filtered_runs.append(run)

    rich_print(f"[bold yellow]Testing {len(filtered_runs)} configurations...[/bold yellow]")
    return filtered_runs


# ----------------------------------------------------------

outputs = remote_parallel_map(train_umc, get_data(), func_cpu=32, detach=True)

rich_print(Panel.fit("[bold green]All seeds processed successfully![/bold green]"))
rich_print(f"[bold cyan]Number of configs tested: {len(outputs)}[/bold cyan]\n")
