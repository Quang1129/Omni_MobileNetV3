import os
import urllib.request
import zipfile
from random import shuffle
from math import floor


def download_dataset(path):
  print("Start downloading TinyImageNet")
  url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  urllib.request.urlretrieve(url, path)


def unzip_data(path):
    path_to_file = path
    directory_extract = os.path.join(os.getcwd(), 'images')
    print("Extracting zip file: %s" % path_to_file)
    with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
      zip_ref.extractall(directory_extract)

    print("Extracted at: %s" % directory_extract)

    return directory_extract


def format_val(val_dir):
    print(f"Formatting: {val_dir}")
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    val_dict = {}
    with open(val_annotations, 'r') as f:
        for line in f:
            line = line.strip().split()
            wnind = line[1]
            image_name = line[0]
            boxes = '\t'.join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((image_name, boxes))

    # print(val_dict)

    for wnind, entries in val_dict.items():
        val_wnid_dir = os.path.join(val_dir, wnind)
        val_images_dir = os.path.join(val_dir, 'images')
        val_wnid_images_dir = os.path.join(val_wnid_dir, 'images')
        os.makedirs(val_wnid_dir, exist_ok = True)
        os.makedirs(val_wnid_images_dir, exist_ok = True)
        wnind_boxes = os.path.join(val_wnid_dir, 'wnind_boxes.txt')

        with open(wnind_boxes, 'w') as f:
            for img_name, box in entries:
                source = os.path.join(val_images_dir, img_name)
                dir = os.path.join(val_wnid_images_dir, img_name)
                os.system(f"cp {source} {dir}")
                f.write(f"{img_name}\t{box}")

            f.close()
            
    os.system(f"rm -rf {val_images_dir}")
    print(f"Cleaning up {val_images_dir}")
    print("Formatting val done")

