from stark_qa.retrieval import STaRKDataset

REGISTERED_DATASETS = ['amazon', 'prime', 'mag']


def load_qa(name: str,
            root: str = 'data/', 
            human_generated_eval: bool = False):

    if name in REGISTERED_DATASETS:
        return STaRKDataset(name, root, 
                            human_generated_eval=human_generated_eval)
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    
    