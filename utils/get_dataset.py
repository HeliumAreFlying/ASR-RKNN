from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    ds =  MsDataset.load(dataset_name='speechoceanadmin/ChineseMandarinSpeechRecognitionCorpus-Mobile',
                         cache_dir="training_cache",
                         subset_name='default',
                         split='train')
    pass