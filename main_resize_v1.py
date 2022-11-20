import argparse
import ast
import numpy as np
from scipy import linalg
import json
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm for progress

import io
from Runtime import *
from utils import *
# from image_dataset_v1 import *
#from image_dataset_v2 import *
import random
from torchvision import transforms

from model_resize_one import *
from image_dataset_resize_v1 import *

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', dest='train_dir', type=str)
    parser.add_argument('-val_dir', dest='val_dir', type=str)
    parser.add_argument('-learning_rate', dest='learning_rate', type=float)
    parser.add_argument('-batch_size', dest='batch_size', type=int)

    parser.add_argument('-num_workers', dest='num_workers', type=int)
    parser.add_argument('-image_width', dest='image_width', type=int)
    parser.add_argument('-image_height', dest='image_height', type=int)
    parser.add_argument('-channels_img', dest='channels_img', type=int)

    parser.add_argument('-l1_lambda', dest='l1_lambda', type=int)
    parser.add_argument('-num_epochs', dest='num_epochs', type=int)
    parser.add_argument('-load_model', dest='load_model', type=bool)
    parser.add_argument('-save_model', dest='save_model', type=bool)

    parser.add_argument('-checkpoint_disc', dest='checkpoint_disc', type=str)
    parser.add_argument('-checkpoint_gen', dest='checkpoint_gen', type=str)
    # parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS']
    # parser.add_argument('-SM_HP_RUNTIME_VAR', dest='SM_HP_RUNTIME_VAR', type = str)

    with open('config.json', 'r') as f:
        parser.set_defaults(**json.load(f))
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args


def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


def train_fn(resizer, loader, optimizer, l1, runtime_log_folder, runtime_log_file_name):
    total_output = ''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        y_fake = resizer(x)

        L1_loss = l1(y_fake, y) * args.l1_lambda

        loss = L1_loss

        loss.backward()
        optimizer.step()

        if idx == (len(loader) - 1):
            print(f'[Epoch {epoch}/{args.num_epochs} (b: {idx})] [Model loss: {loss}]')
            output = f'[Epoch {epoch}/{args.num_epochs} (b: {idx})] [Model loss: {loss}]\n'

            total_output += output

    runtime_log = get_json_file_from_s3(runtime_log_folder, runtime_log_file_name)
    runtime_log += total_output
    upload_json_file_to_s3(runtime_log_folder, runtime_log_file_name, json.dumps(runtime_log))

if __name__ == '__main__':

    args = _parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resizer = Resizer().to(device)
    optimizer = optim.Adam(resizer.parameters(), lr=0.01, betas=(0.5, 0.999))
    L1_LOSS = nn.L1Loss()

    # if model is loaded
    if args.load_model:
        resizer, optimizer = load_checkpoint(checkpoint_filename="resizer.pt",model = resizer,optimizer=optimizer,lr=0.01)

    runtime_dict = ast.literal_eval(os.environ['SM_HP_RUNTIME_VAR'])
    dataset_key = runtime_dict['dataset_key']

    # dataset_name = 'paired-accuracy-speed-modified-v3-dataset'
    # dataset_key = f'datasets/processed-data/{dataset_name}/'

    pair_keys_list = []

    projects = Boto.s3_client.list_objects(Bucket=Boto.bucket_name, Prefix=dataset_key, Delimiter='/')

    for project in projects.get('CommonPrefixes'):

        proj_key = project.get('Prefix')
        pairs = Boto.s3_client.list_objects(Bucket=Boto.bucket_name, Prefix=proj_key, Delimiter='/')
        for pair in pairs.get('CommonPrefixes'):
            pair_key = pair.get('Prefix')

            pair_keys_list.append(pair_key)

    random.shuffle(pair_keys_list)

    train_test_lists = list_splitter(pair_keys_list, 0.8)

    train_list = train_test_lists[0]
    print("Train dataset size: ", len(train_list))
    test_list = train_test_lists[1]
    print("Val dataset size: ", len(test_list))

    transforms = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    ])

    train_dataset = ImageDataset(train_list, transforms)
    val_dataset = ImageDataset(test_list, transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    print("Length train loader ", len(train_loader))

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
    print("Length val loader ", len(val_loader))

    runtime_dict = ast.literal_eval(os.environ['SM_HP_RUNTIME_VAR'])
    log_model_name = runtime_dict['model_name']
    log_dataset_name = runtime_dict['dataset_name']
    log_job_name = runtime_dict['job_name']

    # log_model_name = 'pix2pix-model'
    # log_dataset_name = 'paired-accuracy-speed-modified-v3-dataset'
    # run_datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    # log_job_name = f'{run_datetime}-training'

    runtime_log = ''

    runtime_log_folder = f'training-jobs/{log_model_name}/{log_dataset_name}/{log_job_name}/logs/'
    runtime_log_file_name = 'output_log'
    upload_json_file_to_s3(runtime_log_folder, runtime_log_file_name, json.dumps(runtime_log))

    # for epoch in range(101,args.num_epochs):
    for epoch in range(args.num_epochs):

        train_fn(resizer=resizer,loader=train_loader,optimizer=optimizer,l1=L1_LOSS,runtime_log_folder=runtime_log_folder,runtime_log_file_name=runtime_log_file_name)

        if args.save_model and epoch % 10 == 0:
            save_checkpoint(resizer,optimizer,epoch,"resizer.pt")

        save_some_resize_examples(resizer, val_loader, epoch)



