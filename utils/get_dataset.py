import os
from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    cache_dir = os.path.join(os.getcwd(),"training_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ds = MsDataset.load('modelscope/aishell1_subset',
                        cache_dir=cache_dir,
                        subset_name='default',
                        split='train')