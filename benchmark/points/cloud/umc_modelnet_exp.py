from burla import remote_parallel_map
from rich import print as rich_print
from rich.panel import Panel
import os
import time
import random
import subprocess
import requests
import shutil
import resource

def increase_openfiles_limit(new_value: int):
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_value, hard_limit))  # Increase soft limit

def setup_repo(
    repo_url: str,
    repo_name: str,
    branch_name: str,
    commit_sha: str | None
):
    """Sets up the repository.

    1. Clones the Git repository from the given URL.
    2. Checks out the specified branch.
    3. Changes the working directory to the cloned repository.
    4. Optionally checks out a specific commit.
    """
    time.sleep(random.randint(2, 120))

    os.chdir("/workspace")
    if os.path.exists(repo_name):
        print(f"Repository '{repo_name}' already exists")
    else:
        print(f"Cloning repository '{repo_name}'...")
        subprocess.run(["git", "clone", repo_url])

        os.chdir(repo_name)
        subprocess.run(["git", "checkout", branch_name])
        os.chdir('..')

    os.chdir(repo_name)

    if commit_sha:
        subprocess.run(["git", "checkout", commit_sha])

def upload_file_to_centralfile(file_path, destination_name, project_name):
    """Uploads a file to a project directory in Centralfile.

    Args:
        file_path (str): Path to the file to upload.
        destination_name (str): Name to save the file as in the project.
        project_name (str): Name of the project directory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as file:
        content = file.read()
    api_url = "https://qingy1337--centralfile-flask-app.modal.run/upload"
    payload = {
        "project_name": project_name,
        "file_name": destination_name,
        "content": content
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        print(f"Success! File '{file_path}' uploaded to '{project_name}/{destination_name}'")
        return response.text

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def train_umc(inputs_dict):
    """Trains UMC model from the given config `inputs_dict`."""
    # print(os.listdir('pytorch_geometric/benchmark/data/PointCloud_UMC'))
    # 1. Increase open files limit to avoid CPU nuking the processes
    increase_openfiles_limit(100_000)

    # 2. Setup the repo
    setup_repo(
        repo_url=inputs_dict["repo_url"],
        repo_name=inputs_dict["repo_name"],
        branch_name=inputs_dict["branch_name"],
        commit_sha=inputs_dict["commit_sha"]
    )
    os.chdir("benchmark/points")

    # 3. Run the script with the provided parameters
    print("[Burla] Running with config ---")
    print(inputs_dict)
    print("-------------------------------")
    subprocess.run([
        "python3", "run_all_umc_experiments.py",
        "--dataset", inputs_dict["dataset"],
        "--train_mode", inputs_dict["train_mode"],
        "--lambda_ortho_grid", str(inputs_dict["lambda_ortho_grid"]),
        "--methods", inputs_dict["methods"],
        "--seeds", str(inputs_dict["seeds"]),
    ])

    # Send to Modal
    upload_file_to_centralfile(
        file_path="umc_sweep_results.csv",
        destination_name=f"results_{inputs_dict['dataset']}_{inputs_dict['train_mode']}_{inputs_dict['lambda_ortho_grid']}_{inputs_dict['methods']}_{inputs_dict['seeds']}.csv",
        project_name="modelnet10_umc_burla"
    )

def get_data():
    from burla_io import prepare_inputs

    params_to_test = dict(
        repo_url=[f"https://{os.environ['GITHUB_PAT']}@github.com/cminst/pytorch_geometric.git"],
        repo_name=["pytorch_geometric"],
        branch_name=["main"],
        commit_sha=["8ec4dc781109a4cbb99e80e942f1c31db384c3e5"],
        # ------------------------- Script parameters
        dataset=["ModelNet40"],
        train_mode=["aug","clean"],
        lambda_ortho_grid=[0,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000],
        methods=["umc"],
        seeds=[41,42,43],
    )

    sweep_runs = prepare_inputs(params_to_test)

    # # Filter out already completed runs
    # completed_configs = {
    #     ("ModelNet10", "aug", 0, "umc", 43),
    #     ("ModelNet10", "aug", 0.1, "umc", 44),
    #     ("ModelNet10", "aug", 1, "umc", 45),
    #     ("ModelNet10", "clean", 0, "umc", 41),
    #     ("ModelNet10", "clean", 0.1, "umc", 42),
    #     ("ModelNet10", "clean", 0.0001, "umc", 45),
    #     ("ModelNet10", "clean", 0.001, "umc", 47),
    #     ("ModelNet10", "clean", 1, "umc", 44),
    #     ("ModelNet10", "clean", 1e-05, "umc", 43),
    #     ("ModelNet10", "clean", 10, "umc", 46),
    #     ("ModelNet10", "clean", 1000, "umc", 41),
    # }

    # sweep_runs = [
    #     run for run in sweep_runs
    #     if (run["dataset"], run["train_mode"], run["lambda_ortho_grid"], run["methods"], run["seeds"]) not in completed_configs
    # ]

    rich_print(f"[bold yellow]Testing {len(sweep_runs)} configurations...[/bold yellow]")
    return sweep_runs

# ----------------------------------------------------------

outputs = remote_parallel_map(train_umc, get_data(), func_cpu=64, detach=True)

rich_print(Panel.fit("[bold green]All seeds processed successfully![/bold green]"))
rich_print(f"[bold cyan]Number of configs tested: {len(outputs)}[/bold cyan]\n")
