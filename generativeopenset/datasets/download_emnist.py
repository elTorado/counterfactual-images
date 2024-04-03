import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from scipy import io as sio
import gzip
import struct

DATA_DIR = '/home/deanheizmann/data/emnist/'
DATASET_NAME = 'emnist'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)
IMAGES_LABELS_URL = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
# old path, not working anymore IMAGES_LABELS_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'

#!/usr/bin/env python
# Downloads and processes the EMNIST letters and digits datasets
def download(filename, url):
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping")
    else:
        print(f"Downloading {filename} from {url}...")
        os.system(f'wget -nc {url} -O {filename}')
        if filename.endswith('.zip'):
            os.system('unzip -o *.zip')

def mkdir(path):
    if not os.path.exists(path):
        print(f'Creating directory {path}')
        os.makedirs(path)

import gzip
import os

def extract_gzip(gzip_path, output_path):
    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"The file {output_path} already exists. Extraction skipped.")
        return
    
    print(f"Extracting {gzip_path}...")
    with gzip.open(gzip_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(f_in.read())
    print(f"Extraction completed: {output_path}")


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def convert_emnist(images, labels, fold, category):
    examples = []
    assert images.shape[0] == labels.shape[0], "The number of images and labels must match."

    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for i in tqdm(range(len(images))):
        if category == 'letters':
            # EMNIST letters are offset by 64
            label = chr(labels[i] + 64)
        elif category == 'digits':
            # EMNIST digits are 0-9
            label = str(labels[i])
        
        filename = os.path.join(category_path, f'{category}_{i:06d}.png')
        
        # Check if the image file already exists to avoid re-conversion
        exists=False
        if not os.path.exists(filename):
            # EMNIST images need rotation and reshaping
            image = images[i].reshape(28, 28).transpose()
            Image.fromarray(image, 'L').save(filename)
        else:
            exists = True
        
        examples.append({"filename": filename, "fold": fold, "label": label})
    if exists: print("Some or all files already converted and skipped")
    else:
        print("converted all files, none were skipped")
    return examples

def create_datasets(letters, digits, k = 5000):
    print(" ========= WRITING DATASET FILES ==========")
    #File with all elements
    with open('emnist.dataset', 'w') as fp:
        for element in digits + letters:
            fp.write(json.dumps(element, sort_keys=True) + '\n')
    
    #File with only knowns / digits
    with open('emnist_digits.dataset', 'w') as fp:
        for element in digits:
            fp.write(json.dumps(element, sort_keys=True) + '\n')
    
    #File with only unknowns / letters
    with open('emnist_letters.dataset', 'w') as fp:
        for element in letters:
            fp.write(json.dumps(element, sort_keys=True) + '\n')
            
    #Mixed dataset with 10 digits and 11 letters for comparison with other methods
    with open('emnist_mixed_1to11.dataset', 'w') as fp:
        letters_AtoM = [elem for elem in letters if elem in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M']]
        for element in digits + letters_AtoM:
            fp.write(json.dumps(element, sort_keys=True) + '\n')
    with open('emnist_mixed_2.dataset_12to22', 'w') as fp:
        letters_PtoZ = [elem for elem in letters if elem in ['Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P']]
        for element in digits + letters_PtoZ:
            fp.write(json.dumps(element, sort_keys=True) + '\n') 
           
                    
def main():
    print(f"{DATASET_NAME} dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    download('gzip.zip', IMAGES_LABELS_URL)

    # Process EMNIST letters train set
    extract_gzip('gzip/emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-train-images-idx3-ubyte')
    extract_gzip('gzip/emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte')
    train_images = read_idx('emnist-letters-train-images-idx3-ubyte')
    train_labels = read_idx('emnist-letters-train-labels-idx1-ubyte')
    print("Converting EMNIST letters training set...")
    print("===== Labels =======")
    print(np.unique(train_labels))
    examples_letters = convert_emnist(train_images, train_labels, fold='train', category='letters')
    

    # Process EMNIST digits train set
    extract_gzip('gzip/emnist-digits-train-images-idx3-ubyte.gz', 'emnist-digits-train-images-idx3-ubyte')
    extract_gzip('gzip/emnist-digits-train-labels-idx1-ubyte.gz', 'emnist-digits-train-labels-idx1-ubyte')
    train_images = read_idx('emnist-digits-train-images-idx3-ubyte')
    train_labels = read_idx('emnist-digits-train-labels-idx1-ubyte')
    print("Converting EMNIST digits training set...")
    print("===== Labels =======")
    print(np.unique(train_labels))
    examples_digits = convert_emnist(train_images, train_labels, fold='train', category='digits')

    #create dataset files that contain known and unknown splits
    create_datasets(letters=examples_letters, digits=examples_digits)
if __name__ == '__main__':
    main()