import copy
import json
import multiprocessing
import os
import sys
from functools import partial

import h5py
import numpy as np
import requests
import torch
import torchvision
from PIL import Image
# from tqdm import tqdm
from torch.utils.data import Dataset

import requests
from tqdm import tqdm

# ============ Download UTILS ===============
MIRRORS = [
    'https://zenodo.org/record/4701316/files/%s?download=1',
    'https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/%s'
]

def get_data_path_or_download(dataset, data_root):
    """Finds a dataset locally and downloads if needed.

    Args:
        dataset (str): dataset name. For instance 'camouflage_n=100000_2020-Oct-19.h5py'.
            See https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated for the complete list. (please ignore .a[a-z] extensions)
        data_root (str): path where the dataset will be or is stored. If empty string, it defaults to $TMPDIR 

    Raises:
        ValueError: dataset name does not exist in local path nor in remote

    Returns:
        str: dataset final path 
    """
    if data_root == "":
        data_root = os.environ.get("TMPDIR", "/tmp")
    full_path = os.path.join(data_root, dataset)

    if os.path.isfile(full_path):
        print("%s found." %full_path)
        return full_path
    else:
        print("Downloading %s..." %full_path)

    for i, mirror in enumerate(MIRRORS):
        try:
            download_url(mirror % dataset, full_path)
            return full_path
        except Exception as e:
            if i + 1 < len(MIRRORS):
                print("%s failed, trying %s..." %(mirror, MIRRORS[i+1]))
            else:
                raise e

def download_url(url, path):
    r = requests.head(url)
    if r.status_code == 302:
        raise RuntimeError("Server returned 302 status. \
            Try again later or contact us.")

    is_big = not r.ok
    if is_big:
        r = requests.head(url + ".aa")
        if not r.ok:
            raise ValueError("Dataset %s" %url, "not found in remote.") 
        response = input("Download more than 3GB (Y/N)?: ").lower()
        while response not in ["y", "n"]:
            response = input("Download more than 3GB (Y/N)?: ").lower()
        if response == "n":
            print("Aborted")
            sys.exit(0)
        parts = []
        current_part = "a"
        while r.ok: 
            r = requests.head(url + ".a%s" %current_part)
            parts.append(url + ".a" + current_part)
            current_part = chr(ord(current_part) + 1)
    else:
        parts = [url]

    if not os.path.isfile(path):
        with open(path, 'wb') as file:
            for i, part in enumerate(parts):
                print("Downloading part %d/%d" %(i + 1, len(parts)))
                # Streaming, so we can iterate over the response.
                response = requests.get(part, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kilobyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    file.close()
                    os.remove(path)
                    raise RuntimeError("ERROR, something went wrong downloading %s" %part)

# =====================================================================

class Synbols():
    categorical_attributes = ['char', 'font',
                              'alphabet', 'is_bold', 'is_slant']
    continuous_attributes = ['rotation',
                             'translation.x', 'translation.y', 'scale']

    def __init__(
            self,
            data_path: str,
            task_type: str = "char",
            dataset_name: str = "default_n=100000_2020-Oct-19.h5py",
            download: bool = True):
        """Wraps Synbols in a Pytorch dataset

        Args:
            data_path (str): Path where the dataset will be saved
            task_type (str): Options are 'char', and 'font'
            dataset_name (str): See https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/',
            download (bool): Whether to download the dataset
        """
        if download:  # done here in order to pass x and y to super
            full_path = get_data_path_or_download(dataset_name,
                                                  data_root=data_path)
        else:
            full_path = os.path.join(data_path, dataset_name)

        self.data = SynbolsHDF5(full_path,
                                task_type,
                                mask='random',
                                raw_labels=False)

        self.n_classes = self.data.n_classes

    def get_split(self, split: str, transform):
        """Loads a particular data split

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            torch.utils.data.Dataset: a Pytorch Dataset instance
        """
        return SynbolsSplit(self.data, split, transform=transform)


def _read_json_key(args):
    string, key = args
    return json.loads(string)[key]


def process_task(task, fields):
    data = json.loads(task)
    ret = []
    for field in fields:
        if '.x' in field:
            ret.append(data[field[:-2]][0])
        elif '.y' in field:
            ret.append(data[field[:-2]][1])
        else:
            ret.append(data[field])
    return ret


class SynbolsHDF5:
    """HDF5 Backend Class"""

    def __init__(self, path, task, ratios=[0.6, 0.2, 0.2], mask=None, trim_size=None, raw_labels=False):
        """Constructor: loads data and parses labels.

        Args:
            path (str): path where the data is stored (see full_path above)
            task (str): 'char', 'font', or the field of choice 
            ratios (list, optional): The train/val/test split ratios. Defaults to [0.6, 0.2, 0.2].
            mask (ndarray, optional): Mask with the data partition. Defaults to None.
            trim_size (int, optional): Trim the dataset to a smaller size, for debugging speed. Defaults to None.
            raw_labels (bool, optional): Whether to include all the attributes of the synbols for each batch. Defaults to False.
            reference_mask (ndarray, optional): If train and validation are done with two different datasets, the 
                                                reference mask specifies the partition of the training data. Defaults to None.

        Raises:
            ValueError: Error message
        """
        self.path = path
        self.task = task
        self.ratios = ratios
        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            y = data['y'][...]
            print("Converting json strings to labels...")
            parse_fields = [self.task]
            with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
                self.y = pool.map(
                    partial(process_task, fields=parse_fields), y)
            self.y = list(map(np.array, zip(*self.y)))
            self.y = self.y[0]
            print("Done converting.")

            self.mask = data["split"][mask][...]

            self.raw_labels = None

            self.trim_size = trim_size
            self.reference_mask = None
            print("Done reading hdf5.")
        self.labelset = list(sorted(set(self.y)))
        self.n_classes = len(self.labelset)

    def parse_mask(self, mask, ratios):
        return mask.astype(bool)


class SynbolsSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None):
        """Given a Backend (dataset), it splits the data in train, val, and test.


        Args:
            dataset (object): backend to load, it should contain the following attributes:
                - x, y, mask, ratios, path, task, mask
            split (str): train, val, or test
            transform (torchvision.transforms, optional): A composition of torchvision transforms. Defaults to None.
        """
        self.path = dataset.path
        self.task = dataset.task
        self.mask = dataset.mask
        if dataset.raw_labels is not None:
            self.raw_labelset = dataset.raw_labelset
        self.raw_labels = dataset.raw_labels
        self.ratios = dataset.ratios
        self.labelset = dataset.labelset
        self.split = split
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.split_data(dataset.x, dataset.y, dataset.mask, dataset.ratios)

    def split_data(self, x, y, mask, ratios, rng=np.random.RandomState(42)):
        if mask is None:
            if self.split == 'train':
                start = 0
                end = int(ratios[0] * len(x))
            elif self.split == 'val':
                start = int(ratios[0] * len(x))
                end = int((ratios[0] + ratios[1]) * len(x))
            elif self.split == 'test':
                start = int((ratios[0] + ratios[1]) * len(x))
                end = len(x)
            indices = rng.permutation(len(x))
            indices = indices[start:end]
        else:
            mask = mask[:, ["train", "val", "test"].index(self.split)]
            indices = np.arange(len(y))  # 0....nsamples
            indices = indices[mask]
        self.y = np.array([self.labelset.index(y) for y in y])
        self.x = x[indices]
        self.y = self.y[indices]
        if self.raw_labels is not None:
            self.raw_labels = np.array(self.raw_labels)[indices]

    def __getitem__(self, item):
        if self.raw_labels is None:
            return self.transform(self.x[item]), self.y[item]
        else:
            return self.transform(self.x[item]), self.y[item], self.raw_labels[item]

    def __len__(self):
        return len(self.x)
