# utils

import ast
import os

import torch
# import config
from torchvision.utils import save_image
import io
from Boto import *
from smart_open import open
from Runtime import *
import json


# Method to upload in memory file to s3
def upload_in_mem_file_to_s3(folder_path, file_name_without_extension, in_mem_file, content_type):

    if content_type == 'image/jpeg':
        key = f'{folder_path}{file_name_without_extension}.jpg'


    Boto.s3_client.upload_fileobj(
        in_mem_file,
        Boto.bucket_name,
        key,
        ExtraArgs={'ContentType': content_type}
    )


def get_file_from_filepath(filepath):
    with open(filepath, 'rb') as s3_source:
        return s3_source


def get_json_file_from_s3(folder, file_name_without_extension):
    filepath = f's3://{Boto.bucket_name}/{folder}{file_name_without_extension}.json'

    with open(filepath, 'rb') as s3_source:
        json_file = json.loads(s3_source.read().decode('utf-8'))
        return json_file


def upload_json_file_to_s3(folder, file_name_without_extension, json_file):
    Boto.s3_bucket.put_object(Key=f'{folder}{file_name_without_extension}.json', Body=json_file)


# save images
def save_some_examples(gen, val_loader, epoch):
    runtime_dict = ast.literal_eval(os.environ['SM_HP_RUNTIME_VAR'])
    model_name = runtime_dict['model_name']
    dataset_name = runtime_dict['dataset_name']
    job_name = runtime_dict['job_name']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#

        folder_path = f'training-jobs/{model_name}/{dataset_name}/{job_name}/samples/'

        in_mem_file_y_fake = io.BytesIO()
        save_image(y_fake, in_mem_file_y_fake, format="jpeg")
        in_mem_file_y_fake.seek(0)
        y_fake_filename = f'gen_{epoch}'
        upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=y_fake_filename,
                                 in_mem_file=in_mem_file_y_fake, content_type='image/jpeg')


        in_mem_file_x = io.BytesIO()
        save_image(x * 0.5 + 0.5, in_mem_file_x, format="jpeg")
        in_mem_file_x.seek(0)
        x_filename = f'input_{epoch}'
        upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=x_filename,
                                 in_mem_file=in_mem_file_x, content_type='image/jpeg')


        in_mem_file_y = io.BytesIO()
        save_image(y * 0.5 + 0.5, in_mem_file_y, format="jpeg")
        in_mem_file_y.seek(0)
        y_filename = f'target_{epoch}'
        upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=y_filename,
                                 in_mem_file=in_mem_file_y, content_type='image/jpeg')

        if epoch == 1:
            in_mem_file_y = io.BytesIO()
            save_image(y * 0.5 + 0.5, in_mem_file_y, format='jpeg')
            in_mem_file_y.seek(0)
            y_filename = f'label_{epoch}'
            upload_in_mem_file_to_s3(folder_path=folder_path, file_name_without_extension=y_filename,
                                     in_mem_file=in_mem_file_y, content_type='image/jpeg')


    gen.train()


# save checkpoint
def save_checkpoint(model, optimizer, epoch, filename):
    runtime_dict = ast.literal_eval(os.environ['SM_HP_RUNTIME_VAR'])
    model_name = runtime_dict['model_name']
    dataset_name = runtime_dict['dataset_name']
    job_name = runtime_dict['job_name']
    # print("=> Saving checkpoint")

    folder_path = f'training-jobs/{model_name}/{dataset_name}/{job_name}/checkpoints/E{epoch}/'
    in_mem_file_checkpoint_save = io.BytesIO()

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, in_mem_file_checkpoint_save)

    key = f'{folder_path}{filename}'
    Boto.s3_bucket.put_object(Key=key, Body=in_mem_file_checkpoint_save.getvalue())


# load checkpoint
def load_checkpoint(checkpoint_filename, model, optimizer, lr):
    print("=> Loading checkpoint")
    print("Checkpoint filename: ", checkpoint_filename)
    load_dict = ast.literal_eval(os.environ['SM_HP_LOAD_VAR'])
    model_name = load_dict['load_model_name']
    dataset_name = load_dict['load_dataset_name']
    job_name = load_dict['load_job_name']
    epoch = load_dict['load_epoch']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bucket = 'machine-learning'

    folder_path = f'training-jobs/{model_name}/{dataset_name}/{job_name}/checkpoints/E{epoch}/'
    key = f'{folder_path}{checkpoint_filename}'

    in_mem_file_checkpoint_load = io.BytesIO(Boto.s3_client.get_object(Bucket=bucket, Key=key)['Body'].read())

    checkpoint = torch.load(in_mem_file_checkpoint_load, map_location=device)


    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Finish load checkpoint")

    return model, optimizer

