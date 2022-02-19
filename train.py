from email.policy import default
import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse
import wandb
import uuid

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np
from create_dataset import Dataset
from create_model import SegmentationModel


def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] != 3:
            image = np.argmax(image, axis=-1)

        plt.imshow(image)

    wandb.log({'Validation Segementation': wandb.Image(plt)})
    plt.close()


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
    Init WandB logger
    """
    wandb.init(project="stargazer-segmentation", config=config)

    x_train_dir = os.path.join(args.data_dir, 'train', 'images')
    y_train_dir = os.path.join(args.data_dir, 'train', 'labels')

    x_valid_dir = os.path.join(args.data_dir, 'val', 'images')
    y_valid_dir = os.path.join(args.data_dir, 'val', 'labels')

    """
    Create train and validation dataset
    """
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        preprocessing_fn=smp.encoders.get_preprocessing_fn(
            args.encoder, args.encoder_weights),
        num_classes=args.num_classes
    )

    validation_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        preprocessing_fn=smp.encoders.get_preprocessing_fn(
            args.encoder, args.encoder_weights),
        num_classes=args.num_classes,
        augment=False
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=10)

    valid_loader = DataLoader(validation_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=2)

    """
    Create the model
    """
    model = SegmentationModel(args.encoder,
                              args.encoder_weights,
                              args.num_classes,
                              args.activation,
                              args.device,
                              logdir
                              )

    if args.checkpoint_dir is not None:
        model.load_checkpoint(args.checkpoint_dir)

    """
    Train the model
    """
    model.train(train_loader,
                valid_loader,
                args.epochs,
                args.lr)

    """
    Visualise model performance
    """
    for i in range(5):
        n = np.random.choice(len(validation_dataset))

        image, gt_mask = validation_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(args.device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )


if __name__ == "__main__":
    main()
