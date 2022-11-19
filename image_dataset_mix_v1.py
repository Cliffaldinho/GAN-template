import math
import urllib3
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import boto3
from smart_open import open
import io
from pathlib import Path
from Boto import *
import random
from torch.utils.data import *
from torchvision import transforms
from torchvision import transforms as T
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ConnectionError
from urllib3.exceptions import HTTPError
import time
import cv2

from botocore.exceptions import ClientError


# google maps dataset
class ImageDataset(Dataset):

    # send in root directory
    def __init__(self, list_files, transform, s3_client=None):

        """
        bucket_name = 'istu-ml-training'
        s3_resource = boto3.resource('s3')
        s3_client = boto3.client('s3')

        # note: assumption that bucket is always istu-ml-training
        s3_bucket = s3_resource.Bucket(bucket_name)
        """
        #    pass

        # def open_root_dir(self, root_dir):

        # root directory
        # self.root_dir = root_dir

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
        # print("list files")
        # print(self.list_files)
        self.list_files = list_files
        print("list files length")
        print(len(self.list_files))
        # self.bucket_name = 'istu-ml-training'

        # self.s3_resource = boto3.resource('s3')
        # self.s3_bucket = self.s3_resource.Bucket(self.bucket_name)
        # self.s3_client = boto3.client('s3')

        # self.train_dir = f'datasets/training-data/maps/train/'

        # list all of the files in there
        # self.list_files = self.get_files_in_s3(self.root_dir)
        # self.list_files = os.listdir(self.root_dir)

        # because maps before and after is together
        # just need to split it by half
        # print(self.root_dir)
        # print(self.list_files)
        # print(self.__len__())

    def get_file_from_filepath(self, filepath):
        with open(filepath, 'rb') as s3_source:
            return s3_source

    # length of dataset
    def __len__(self):
        return len(self.list_files)

    """
    def get_files_in_s3(self, folder_key):
        files = [f.key.split(f'{folder_key}')[-1]
                 for f in Boto.s3_bucket.objects.filter(Prefix=folder_key).all()
                 if f.key[-1] != "/"]
        return files

    def smart_open_file(self, filepath):
        with open(filepath, 'rb') as s3_source:
            return s3_source
    """
    """
    def worker_init_fn(self,worker_id):

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_end = dataset.end
        # configure the dataset to only process the split workload
        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)
    """

    # get item
    def __getitem__(self, index):
        # got_item = False

        input_image = None
        target_image = None
        # concat_image = None
        # cropped_concat_image = None
        # input_split = None
        # target_split = None
        while input_image is None or target_image is None:
            try:

                # print("Flag start")
                # print("Item Index ",index)
                # print("Worker Info ", get_worker_info())

                pair_key = self.list_files[index]
                # print("Pair key ",pair_key)
                # pair = Boto.s3_client.list_objects(Bucket=Boto.bucket_name, Prefix=pair_key, Delimiter='/')
                pair = self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=pair_key, Delimiter='/')

                input_image_key = pair.get('Contents')[1].get('Key')
                # input_image_path = f's3://{Boto.bucket_name}/{input_image_key}'
                input_image_path = f's3://{self.bucket_name}/{input_image_key}'
                # print("Input image path ",input_image_path)

                input_image_s3_source = self.get_file_from_filepath(input_image_path)
                pil_input_image = Image.open(input_image_s3_source)

                # input_image = np.array()
                # print("Input image shape ",np.shape(input_image))

                target_image_key = pair.get('Contents')[0].get('Key')
                # target_image_path = f's3://{Boto.bucket_name}/{target_image_key}'
                target_image_path = f's3://{self.bucket_name}/{target_image_key}'
                # print("Target image path ",target_image_path)

                target_image_s3_source = self.get_file_from_filepath(target_image_path)
                pil_target_image = Image.open(target_image_s3_source)

                """
                #Test if images are same size
                if pil_input_image.size != pil_target_image.size:
                    print("Input and target original image size not the same.")
                    print("Input size: ",pil_input_image.size)
                    print("Target size: ",pil_target_image.size)
                    print()
                else:
                    print("Input target same size.")
                    print()
                """

                """
                # Align images
                input_and_target_images = [pil_input_image,pil_target_image]
                alignMTB = cv2.createAlignMTB()
                alignMTB.process(input_and_target_images,input_and_target_images)
                pil_input_image = input_and_target_images[0]
                pil_target_image = input_and_target_images[1]
                """

                input_image = self.transform(pil_input_image)
                target_image = self.transform(pil_target_image)

                """
                # Crop images 
                input_tensor = self.transform(pil_input_image)
                target_tensor = self.transform(pil_target_image)

                concat_image = torch.cat([input_tensor, target_tensor], dim=0)

                crop_transform = T.RandomCrop(256)
                cropped_concat_image = crop_transform(concat_image)
                input_image, target_image = torch.split(cropped_concat_image,3,dim=0)
                """

                """
                # Align images
                input_and_target_images = [input_image,target_image]
                alignMTB = cv2.createAlignMTB()
                alignMTB.process(input_and_target_images,input_and_target_images)
                input_image = input_and_target_images[0]
                target_image = input_and_target_images[1]
                """

                # augmentations = config.both_transform(image=input_image, image0=target_image)

                # get input image and target image by doing augmentations of images
                # input_image, target_image = augmentations['image'], augmentations['image0']

                # input_image = config.transform_only_input(image=input_image)['image']
                # target_image = config.transform_only_mask(image=target_image)['image']


            except (ConnectionError, ProtocolError, HTTPError) as e:

                print(f'Caught Error {e}, try again')
                time.sleep(1)

            else:
                return input_image, target_image


if __name__ == "__main__":
    # run_key = f'datasets/processed-data/2022-05-09-14-39-18-dataset/match-raws-finals/U12239/'
    run_key = f'datasets/processed-data/paired-accuracy-speed-modified-v2-dataset/'
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
    train_loader = DataLoader(train_dataset, batch_size=8)
    counter = 0
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        break
    """
    for x, y, z, d,e,f in train_loader:
        counter+=1

        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(d.shape)
        print(e.shape)
        print(f.shape)

        save_image(e, f'save_data/x{counter}.jpg')
        save_image(f, f'save_data/y{counter}.jpg')

        if counter == 2:
            break
    """
    """

    train_dataset = ImageDataset(train_list)
    #from MultiDataLoader import *
    #train_loader = MultiProcessDataLoader(train_dataset,batch_size=32,num_workers=2)
    train_loader = DataLoader(train_dataset,batch_size=32,num_workers=2)
    print("Length loader: ",len(train_loader))
    counter_list = [0,1,2]
    for counter in counter_list:
        print("Counter: ",counter)
    #print("Length loader: ", train_loader.__sizeof__())
        for index,(x,y) in enumerate(train_loader):
            print("Loader Index: ",index)
        #save_image(x, f'save_data/x{index}.jpg')
        #save_image(y,f'save_data/y{index}.jpg')
        #if index == 10:
        #    break


    #dataset = ImageDataset(root_dir=root_dir)
    """
    """
    train_address = 'C://Users/Cliff/Desktop/ImageCloud/Feature_Store/data/maps/train'
    train_dir = f'datasets/training-data/maps/train/'
    dataset = MapDataset(train_dir)
    loader = DataLoader(dataset, batch_size=1)

    counter = 0
    # print(len(loader))

    for x, y in loader:
        # print(x.dtype)
        # print(y.shape)

        img_filename = Path(dataset.list_files[counter]).stem
        # print(img_filename)
        folder_path = 'datasets/training-data/maps/test/'
        x_filename = 'input' + '_' + img_filename
        y_filename = 'target' + '_' + img_filename

        in_mem_file_x = io.BytesIO()
        save_image(x, in_mem_file_x, format="jpeg")
        in_mem_file_x.seek(0)
        dataset.upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=x_filename,
                                         in_mem_file=in_mem_file_x)

        in_mem_file_y = io.BytesIO()
        save_image(y, in_mem_file_y, format="jpeg")
        in_mem_file_y.seek(0)
        dataset.upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=y_filename,
                                         in_mem_file=in_mem_file_y)

        # save_image(y, f'save_data/y{counter}.jpeg')

        counter += 1

        print(counter)
        if counter == 11:
            break

        # import sys
        # sys.exit()
    """




