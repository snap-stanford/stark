import os
import os.path as osp
import pickle
import torch
import json


def read_from_file(file_path):
    if '.txt' in file_path:
        with open(file_path, 'r') as f:
            return f.read()
    elif '.json' in file_path:
        with open(file_path, 'r') as f:
            return json.load(f)
    elif '.pkl' in file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise NotImplementedError(f'File type not supported: {file_path}')


def write_to_file(file_path, content):
    if '.txt' in file_path:
        with open(file_path, 'w') as f:
            f.write(content)
    elif '.json' in file_path:
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=4)
    elif '.pkl' in file_path:
        with open(file_path, 'wb') as f:
            pickle.dump(content, f)
    else:
        raise NotImplementedError(f'File type not supported: {file_path}')


def save_files(save_path, **kwargs):
    os.makedirs(save_path, exist_ok=True)
    for key, value in kwargs.items():
        if isinstance(value, dict):
            with open(osp.join(save_path, f'{key}.pkl'), 'wb') as f:
                pickle.dump(value, f)
        elif isinstance(value, torch.Tensor):
            torch.save(value, osp.join(save_path, f'{key}.pt'))
        else:
            pass
        
        
def load_files(save_path):
    loaded_dict = {}
    for file in os.listdir(save_path):
        if os.path.isdir(osp.join(save_path, file)): 
            continue
        if file.endswith('.pkl'):
            with open(osp.join(save_path, file), 'rb') as f:
                loaded_dict[file.split('.')[0]] = pickle.load(f)
        elif file.endswith('.pt'):
            loaded_dict[file.split('.')[0]] = torch.load(osp.join(save_path, file))
        else:
            pass
    return loaded_dict 
        
    