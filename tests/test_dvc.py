import os
import subprocess

def test_dvc_setup():
    """
    Ensure the .dvc directory exists and DVC is functional.
    """
    repo_root = os.getcwd()
    while repo_root != "../":
        if os.path.isdir(os.path.join(repo_root, ".dvc")):
            break
        repo_root = os.path.dirname(repo_root)
    else:
        assert False, "❌ .dvc directory not found."

    try:
        subprocess.run(["dvc", "status"], cwd=repo_root, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert False, f"❌ DVC command failed: {e.stderr.decode()}"