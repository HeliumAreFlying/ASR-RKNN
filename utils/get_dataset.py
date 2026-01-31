import os
from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    cache_dir = os.path.join(os.getcwd(), 'training_cache')
    ds =  MsDataset.load(dataset_name='speechoceanadmin/ChineseMandarinSpeechRecognitionCorpus-Mobile',
                         cache_dir=cache_dir,
                         subset_name='default',
                         split='train')
    pass