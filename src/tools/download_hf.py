import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files


def download_hf_file(repo, file, repo_type="dataset", save_as_file=None):
    """
    Downloads a file from a Hugging Face repository and saves it to the specified path.

    Args:
        repo (str): The repository name.
        file (str): The file path within the repository to download.
        repo_type (str): The type of the repository (e.g., 'dataset').
        save_as_file (str, optional): The local file path to save the downloaded file. If not provided, 
                                      saves the file in the current directory with the same name as the 
                                      original file.
    """
    file_path = hf_hub_download(repo, file, repo_type=repo_type)
    if save_as_file is None:
        save_as_file = os.path.basename(file)
    
    os.makedirs(os.path.dirname(save_as_file), exist_ok=True)
    shutil.copy2(file_path, save_as_file)  # Use copy2 instead of move

    print(f"Downloaded <file:{file}> from <repo:{repo}> to <path:{save_as_file}>!")



def download_hf_folder(repo, folder, repo_type="dataset", save_as_folder="data/"):
    """
    Downloads a folder from a Hugging Face repository and saves it to the specified directory.

    Args:
        repo (str): The repository name.
        folder (str): The folder path within the repository to download.
        repo_type (str): The type of the repository (e.g., 'dataset').
        save_as_folder (str, optional): The local directory to save the downloaded folder. Defaults to "data/".
    """
    files = list_repo_files(repo, repo_type=repo_type)
    folder_files = [f for f in files if f.startswith(folder + '/')]
    for file in folder_files:
        file_path = hf_hub_download(repo, file, repo_type=repo_type)
        new_file_path = os.path.join(save_as_folder, os.path.relpath(file, folder))
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        shutil.copy2(file_path, new_file_path)  # Use copy2 instead of move
        
    print(f"Downloaded <folder:{folder}> with {len(folder_files)} files from <repo:{repo}> to <path:{save_as_folder}>!")

