import os
from cv2 import merge
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse
import wandb
import uuid
import json

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np
from segmentation.dataset import Dataset
from segmentation.model import SegmentationModel
from segmentation_models_pytorch.encoders._preprocessing import preprocess_input
import functools
from convert_to_coreml import create_coreml_model


def visualize(i, mean, std, num_classes, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if image.shape[0] != 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.argmax(image, axis=-1)
            plt.imshow(image, vmin=0, vmax=num_classes)
        else:
            image = (np.transpose(image, (1, 2, 0)) + (mean/std)) / (1/std)
            plt.imshow(image)

    wandb.log({'eval/it': i,
               'eval/segmentation': wandb.Image(plt)})
    plt.close()


"""
CoreML only supports equal scaling for every channel.
Make sure the training params can be converted to CoreML later.
"""


def center_preprocessing_function(preprocessing_params):
    preprocessing_params['std'] = [np.mean(preprocessing_params['std'])] * 3
    preprocessing_params['mean'] = [np.mean(preprocessing_params['mean'])] * 3

    mean = np.mean(preprocessing_params['mean'])
    std = np.mean(preprocessing_params['std'])
    return functools.partial(preprocess_input, **preprocessing_params), mean, std


def create_dataloaders(location, preprocess_fn, config):
    x_train_dir = os.path.join(location, 'train', 'images')
    y_train_dir = os.path.join(location, 'train', 'labels')

    x_valid_dir = os.path.join(location, 'val', 'images')
    y_valid_dir = os.path.join(location, 'val', 'labels')

    """
    Create train and validation dataset
    """
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        preprocessing_fn=preprocess_fn,
        num_classes=config['num_classes'],
        merge_classes=config['merge_classes']
    )

    validation_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        preprocessing_fn=preprocess_fn,
        num_classes=config['num_classes'],
        augment=False,
        merge_classes=config['merge_classes']
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=10)

    valid_loader = DataLoader(validation_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=2)

    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    parser.add_argument("--data_dir", type=str,
                        help='Expects a data dir with subfolders train/, val/ with subfolders images/, labels/ containing annotated pairs')

    parser.add_argument("--save_dir", type=str,
                        help='Directory to save the trained model to')

    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help='If set, the model will bo loaded from this state.')

    parser.add_argument("--num_classes", type=int,
                        help='Number of classes in the dataset including background')

    parser.add_argument("--architecture", default="fpn", type=str,
                        help='Model architecture. Available: fpn, deeplab, unet')

    parser.add_argument("--encoder", default="se_resnext50_32x4d", type=str,
                        help='Segementation encoder')

    parser.add_argument("--encoder_weights", default="imagenet", type=str,
                        help='Encoder initalisation')

    parser.add_argument("--activation", default="softmax2d", type=str,
                        help='Activation fucntion')

    parser.add_argument("--device", default="cpu", type=str,
                        help='cuda or cpu')

    parser.add_argument("--epochs", default=10, type=int,
                        help='Number of epochs to train')

    parser.add_argument("--lr", default=0.0001, type=float,
                        help='Inital learning rate')

    parser.add_argument("--num_plots", default=10, type=int,
                        help='Number of visulistations of model performance after training.')

    parser.add_argument("--merge_classes", default=None, type=json.loads,
                        help='Specify classes to be merged. Pass list of lists. Classes inside the same sublist will be merged into one.')

    parser.add_argument("--pretrain_dir", type=str, default=None,
                        help='Directory of the prettraining dataset.')

    parser.add_argument("--pretrain_epochs", default=100, type=int,
                        help='Number of epochs to train on pretrain dataset.')

    args = parser.parse_args()

    """
    Create config dict from parser args
    """
    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]

    identifier = uuid.uuid1()
    logdir = os.path.join(config['save_dir'], 'saved_models', str(identifier))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config['logdir'] = logdir

    """
    Create CoreML compatible scaler
    """
    params = smp.encoders.get_preprocessing_params(
        args.encoder, args.encoder_weights)

    preprocessing_function, mean, std = center_preprocessing_function(params)
    config['mean'] = mean
    config['std'] = std

    """
    Init WandB logger
    """
    wandb.init(project="stargazer-segmentation", config=config)
    wandb.define_metric(name="train/epoch")
    wandb.define_metric(name="train/*",
                        step_metric="train/epoch")

    wandb.define_metric(name="eval/it")
    wandb.define_metric(name="eval/*",
                        step_metric="eval/it")

    """
    Create the model
    """
    num_classes = args.num_classes if args.merge_classes is None else len(
        args.merge_classes)
    model = SegmentationModel(args.architecture,
                              args.encoder,
                              args.encoder_weights,
                              num_classes,
                              args.activation,
                              args.device,
                              logdir,
                              config
                              )

    if args.checkpoint_dir is not None:
        model.load_checkpoint(args.checkpoint_dir)

    """
    Pretrain the model if dataset provided
    """
    if args.pretrain_dir != None:
        train_loader, valid_loader = create_dataloaders(
            args.pretrain_dir, preprocessing_function, config)

        model.train(train_loader,
                    valid_loader,
                    args.epochs,
                    args.lr)

    """
    Train the model
    """
    train_loader, valid_loader = create_dataloaders(
        args.data_dir, preprocessing_function, config)

    model.train(train_loader,
                valid_loader,
                args.pretrain_epochs,
                args.lr)

    """
    Export coreML model to WandB
    """
    create_coreml_model(
        model_dir=os.path.join(logdir, "model_latest.pth"),
        out_dir=os.path.join(logdir, "segmentation.mlmodel"),
        mean=mean,
        std=std)

    wandb.save(os.path.join(logdir, "segmentation.mlmodel"))
    


if __name__ == "__main__":
    main()
