#!/usr/bin/env python
# Downloads the EMNIST letters dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from scipy import io as sio

DATA_DIR = '/home/deanheizmann/data/emnist/matlab.zip'
DATASET_NAME = 'emnist'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

# Alternative format matching the original MNIST
IMAGES_LABELS_URL = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'

# old path, not working anymore IMAGES_LABELS_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'




def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        print(f"Downloading {filename} from {url}...")
        os.system(f'wget -nc {url} -O {filename}')
        if filename.endswith('.zip'):
            os.system('unzip -o *.zip')

def mkdir(path):
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.makedirs(path)

def extract_gzip(gzip_path, output_path):
    print(f"Extracting {gzip_path}...")
    with gzip.open(gzip_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(f_in.read())

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def convert_emnist(images, labels, fold):
    examples = []
    assert images.shape[0] == labels.shape[0]
    letters_path = os.path.join(DATASET_PATH, 'letters')
    mkdir(letters_path)
    for i in tqdm(range(len(images))):
        label = chr(labels[i] + 64)  # Adjust label calculation as needed
        filename = os.path.join(letters_path, f'letter_{i:06d}.png')
        image = images[i].reshape(28, 28)  # Adjust reshape for your dataset's format
        Image.fromarray(image).save(filename)
        examples.append({"filename": filename, "fold": fold, "label": label})
    return examples

def main():
    print(f"{DATASET_NAME} dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    download('gzip.zip', IMAGES_LABELS_URL)

    # Process example for EMNIST letters train set
    extract_gzip('gzip/emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-train-images-idx3-ubyte')
    extract_gzip('gzip/emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte')

    train_images = read_idx('emnist-letters-train-images-idx3-ubyte')
    train_labels = read_idx('emnist-letters-train-labels-idx1-ubyte')

    print("Converting EMNIST letters training set...")
    train_examples = convert_emnist(train_images, train_labels, fold='train')

    # Repeat for test set and any other sets as needed

    print("Dataset conversion finished")

if __name__ == '__main__':
    main()