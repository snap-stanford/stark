import os
import os.path as osp
import pickle
import torch
import json


def read_from_file(file_path):
    """
    Read content from a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        content: Content of the file.

    Raises:
        NotImplementedError: If the file type is not supported.
    """
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise NotImplementedError(f'File type not supported: {file_path}')


def write_to_file(file_path, content):
    """
    Write content to a file based on its extension.

    Args:
        file_path (str): Path to the file.
        content: Content to write.

    Raises:
        NotImplementedError: If the file type is not supported.
    """
    if file_path.endswith('.txt'):
        with open(file_path, 'w') as f:
            f.write(content)
    elif file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=4)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(content, f)
    else:
        raise NotImplementedError(f'File type not supported: {file_path}')


def save_files(save_path, **kwargs):
    """
    Save multiple files in a specified directory.

    Args:
        save_path (str): Directory to save the files.
        **kwargs: Keyword arguments where keys are filenames (without extension) and values are the contents.
    """
    os.makedirs(save_path, exist_ok=True)
    for key, value in kwargs.items():
        if isinstance(value, dict):
            with open(osp.join(save_path, f'{key}.pkl'), 'wb') as f:
                pickle.dump(value, f)
        elif isinstance(value, torch.Tensor):
            torch.save(value, osp.join(save_path, f'{key}.pt'))
        else:
            raise NotImplementedError(f'File type not supported for key: {key}')


def load_files(save_path):
    """
    Load all files from a specified directory.

    Args:
        save_path (str): Directory to load the files from.

    Returns:
        dict: Dictionary with filenames (without extension) as keys and file contents as values.
    """
    loaded_dict = {}
    for file in os.listdir(save_path):
        if os.path.isdir(osp.join(save_path, file)):
            continue
        file_path = osp.join(save_path, file)
        file_name, file_ext = osp.splitext(file)
        if file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                loaded_dict[file_name] = pickle.load(f)
        elif file_ext == '.pt':
            loaded_dict[file_name] = torch.load(file_path)
        else:
            raise NotImplementedError(f'File type not supported: {file}')
    return loaded_dict
