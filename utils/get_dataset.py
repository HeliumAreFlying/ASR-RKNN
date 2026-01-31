from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    ds =  MsDataset.load('speechoceanadmin/ChineseMandarinSpeechRecognitionCorpus-Mobile', subset_name='default', split='train')
    