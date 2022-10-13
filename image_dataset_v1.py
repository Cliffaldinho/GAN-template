# Dataset class

import math
import urllib3
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import boto3
# import config
from smart_open import open
import io
from pathlib import Path
from Boto import *
import random
# from utils import *
from torch.utils.data import *
from torchvision import transforms
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ConnectionError
from urllib3.exceptions import HTTPError
import time

from botocore.exceptions import ClientError


# google maps dataset
class ImageDataset(Dataset):

    # send in root directory
    def __init__(self, list_files, transform, s3_client=None):


        print("Initialize dataset")
        self.transform = transform
        if s3_client == None:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
        print("s3 client")
        print(self.s3_client)
        self.bucket_name = 'istu-ml-training'
        print("bucket name")
        print(self.bucket_name)

        self.list_files = list_files
        print("list files length")
        print(len(self.list_files))

    def get_file_from_filepath(self, filepath):
        with open(filepath, 'rb') as s3_source:
            return s3_source

    # length of dataset
    def __len__(self):
        return len(self.list_files)

    # get item
    def __getitem__(self, index):
        # got_item = False

        input_image = None
        target_image = None
        while input_image is None or target_image is None:
            try:

                pair_key = self.list_files[index]
                pair = self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=pair_key, Delimiter='/')

                input_image_key = pair.get('Contents')[1].get('Key')
                input_image_path = f's3://{self.bucket_name}/{input_image_key}'

                input_image_s3_source = self.get_file_from_filepath(input_image_path)
                pil_input_image = Image.open(input_image_s3_source)

                target_image_key = pair.get('Contents')[0].get('Key')
                target_image_path = f's3://{self.bucket_name}/{target_image_key}'

                target_image_s3_source = self.get_file_from_filepath(target_image_path)
                pil_target_image = Image.open(target_image_s3_source)

                input_image = self.transform(pil_input_image)
                target_image = self.transform(pil_target_image)

            except (ConnectionError, ProtocolError, HTTPError) as e:

                print(f'Caught Error {e}, try again')
                time.sleep(1)

            else:
                return input_image, target_image


if __name__ == "__main__":
    # run_key = f'datasets/processed-data/2022-05-09-14-39-18-dataset/match-raws-finals/U12239/'
    run_key = f'datasets/processed-data/paired-accuracy-speed-modified-v1-dataset/'
    # bucket = 'istu-ml-training'

    pair_keys_list = []

    projects = Boto.s3_client.list_objects(Bucket=Boto.bucket_name, Prefix=run_key, Delimiter='/')
    # print(projects)
    for project in projects.get('CommonPrefixes'):

        # print()
        # print('*'*100)
        proj_key = project.get('Prefix')
        # print(proj_key)
        pairs = Boto.s3_client.list_objects(Bucket=Boto.bucket_name, Prefix=proj_key, Delimiter='/')
        for pair in pairs.get('CommonPrefixes'):
            pair_key = pair.get('Prefix')
            # print(pair_key)
            pair_keys_list.append(pair_key)

    print(len(pair_keys_list))

    random.shuffle(pair_keys_list)


    # selected_list = random.sample(pair_keys_list, 700)
    # print(len(selected_list))

    def list_splitter(list_to_split, ratio):
        elements = len(list_to_split)
        middle = int(elements * ratio)
        return [list_to_split[:middle], list_to_split[middle:]]


    train_test_lists = list_splitter(pair_keys_list, 0.8)

    train_list = train_test_lists[0]
    test_list = train_test_lists[1]

    print(len(train_list))
    print(len(test_list))

    transforms = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    ])

    train_dataset = ImageDataset(train_list, transforms)
    train_loader = DataLoader(train_dataset, batch_size=4)
    counter = 0
    for x, y in train_loader:
        counter += 1

        print(x.shape)
        print(y.shape)

        save_image(x, f'save_data/x{counter}.jpg')
        save_image(y, f'save_data/y{counter}.jpg')

        if counter == 5:
            break



