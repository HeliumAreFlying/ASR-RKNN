import os

def get_filepaths(directory,extension="txt",print_size=True):
    filepaths = []
    for root,dirs,files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root,_file))
    if print_size:
        print("size of filepaths = ",len(filepaths))
    return filepaths