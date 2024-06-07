from src.benchmarks.qa_datasets import STaRKDataset

REGISTERED_DATASETS = ['amazon', 'primekg', 'mag']


def get_qa_dataset(name: str,
                   root: str = 'data/', 
                   human_generated_eval: bool = False):

    if name in REGISTERED_DATASETS:
        return STaRKDataset(name, root, 
                            human_generated_eval=human_generated_eval)
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    