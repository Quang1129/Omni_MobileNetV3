import os
from .downloading import *
from .data_transform import *
from .tiny_imagenet import * 

if not os.path.exists('./images'):
    path = "./tiny-imagenet-200"
    
    download_dataset(path)
    unzip_path = unzip_data(path)
    
    val_dir = os.path.join(unzip_path, 'tiny-imagenet-200/val')
    
    format_val(val_dir)
else:
    print("The Dataset is already downloaded")