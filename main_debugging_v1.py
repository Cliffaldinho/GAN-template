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
from ImageDataset import *
import random
from torchvision import transforms
from DayDuskDataset import *

# from generator_model import Generator # import generator
# from discriminator_model import Discriminator # import discriminator
# from generator_model_control import Generator
# from discriminator_model_control import Discriminator
# from generator_model_features import Generator
# from discriminator_model_features import Discriminator
# from generator_model_layers import Generator
# from discriminator_model_layers import Discriminator
# from generator_model_features_and_layers import Generator
# from discriminator_model_features_and_layers import Discriminator
# from generator_model_identity import Generator
# from discriminator_model_identity import Discriminator

# from generator_model_residuals import Generator
# from discriminator_model_residuals import Discriminator
# from generator_model_stride_2_kernel_6 import Generator
# from generator_model_gaussian_layer_v3 import Generator
# from generator_model_control import Generator
# from generator_model_layer1024_control import Generator
# from generator_model_layer1024_control import Generator
# from generator_model_layer512_control import Generator
# from generator_model_layer512_replacement import Generator
# from discriminator_model_control import Discriminator

# from generator_model_replacement import Generator
# from discriminator_model_reduce_downsampling import Discriminator
# from generator_model_control import Generator
# from discriminator_model_control import Discriminator

from generator_model_debugging import Generator
from discriminator_model_debugging import Discriminator


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


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, runtime_log_folder, runtime_log_file_name):
    total_output = ''

    #loop = tqdm(loader, leave=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #for idx, (x, y) in enumerate(loop):
    for idx, (x,y) in enumerate(loader):

        # train discriminator
        opt_disc.zero_grad()

        x = x.to(device)
        y = y.to(device)

        y_fake = gen(x)

        disc_input_real = torch.cat([x, y], dim=1)
        D_real = disc(disc_input_real)

        disc_input_fake = torch.cat([x, y_fake], dim=1)
        D_fake = disc(disc_input_fake.detach())

        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

        D_loss = (D_real_loss + D_fake_loss) / 2

        D_loss.backward()
        opt_disc.step()

        # train generator

        opt_gen.zero_grad()
        fake_gen = torch.cat([x, y_fake], dim=1)
        D_fake = disc(fake_gen)

        # compute fake loss
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

        # compute L1 loss
        L1 = l1(y_fake, y) * args.l1_lambda

        G_loss = G_fake_loss + L1
        G_loss = L1

        G_loss.backward()
        opt_gen.step()

        #if idx == (len(loop) - 1):
        if idx == (len(loader) - 1):
            print(
                f'[Epoch {epoch}/{args.num_epochs} (b: {idx})] [D loss: {D_loss}, D real loss: {D_real_loss}, D fake loss: {D_fake_loss}] [G loss: {G_loss}, G fake loss: {G_fake_loss}, L1 loss: {L1}]')

            output = f'[Epoch {epoch}/{args.num_epochs} (b: {idx})] [D loss: {D_loss}, D real loss: {D_real_loss}, D fake loss: {D_fake_loss}] [G loss: {G_loss}, G fake loss: {G_fake_loss}, L1 loss: {L1}]\n'
            total_output += output

    runtime_log = get_json_file_from_s3(runtime_log_folder, runtime_log_file_name)
    runtime_log += total_output
    upload_json_file_to_s3(runtime_log_folder, runtime_log_file_name, json.dumps(runtime_log))


if __name__ == '__main__':

    args = _parse_args()

    # create discriminator and generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc = Discriminator(in_channels=3).to(device)
    gen = Generator(in_channels=3).to(device)

    # initialize optimizer
    opt_disc = optim.Adam(disc.parameters(), lr=args.learning_rate,
                          betas=(0.5, 0.999))  # betas for Adam optimizer as per paper
    opt_gen = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # BCE loss - binary cross entropy + sigmoid function = BCE with logits
    BCE = nn.BCEWithLogitsLoss()

    # L1 loss
    L1_LOSS = nn.L1Loss()

    # if model is loaded
    if args.load_model:
        # load checkpoint
        gen, opt_gen = load_checkpoint(args.checkpoint_gen, gen, opt_gen, args.learning_rate)
        disc, opt_disc = load_checkpoint(args.checkpoint_disc, disc, opt_disc, args.learning_rate)

    runtime_dict = ast.literal_eval(os.environ['SM_HP_RUNTIME_VAR'])
    dataset_key = runtime_dict['dataset_key']

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
        transforms.Resize((256, 256)),
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

    runtime_log = ''

    runtime_log_folder = f'training-jobs/{log_model_name}/{log_dataset_name}/{log_job_name}/logs/'
    runtime_log_file_name = 'output_log'
    upload_json_file_to_s3(runtime_log_folder, runtime_log_file_name, json.dumps(runtime_log))

    for epoch in range(args.num_epochs):

        train_fn(disc=disc, gen=gen, loader=train_loader, opt_disc=opt_disc, opt_gen=opt_gen, l1=L1_LOSS, bce=BCE,
                 runtime_log_folder=runtime_log_folder, runtime_log_file_name=runtime_log_file_name)

        if args.save_model and epoch % 10 == 0:
            save_checkpoint(gen, opt_gen, epoch, args.checkpoint_gen)
            save_checkpoint(disc, opt_disc, epoch, args.checkpoint_disc)

        save_some_examples(gen, val_loader, epoch)



