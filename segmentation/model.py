from distutils.command.config import config
from cv2 import mean
import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
import wandb
import matplotlib.pyplot as plt
import wandb


class SegmentationModel:

    def __init__(self, architecture,  encoder, encoder_weights, num_classes, activation, device, save_dir, config) -> None:
        print("Init with classes: ", num_classes)

        # create segmentation model with pretrained encoder
        if architecture == 'fpn':
            self.model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                activation=activation,
            )
        elif architecture == 'deeplab':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                encoder_output_stride=8,
                classes=num_classes,
                activation=activation,
            )
        elif architecture == 'unet':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=num_classes,
                activation=activation,
            )
        else:
            raise "Architecture not in list"

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)

        self.device = device
        self.save_dir = save_dir

        self.config = config

    def load_checkpoint(self, path):
        self.model = torch.load(path)

    def visualize(epoch, mean, std, num_classes, **images):
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

        wandb.log({
            'train/epoch': epoch,
            'train/validation/segmentation': wandb.Image(plt)}
        )
        plt.close()

    def train(self,
              train_loader,
              val_loader,
              epochs=10,
              lr=0.0001,
              plot_interval=1):
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=lr),
        ])

        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=loss,
            metrics=metrics,
            device=self.device,
            verbose=True,
        )

        max_score = 0
        for i in range(epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            wandb.log({'train/epoch': i,
                       'train/train/dice_loss': train_logs['dice_loss'],
                       'train/train/iou_score': train_logs['iou_score']})

            wandb.log({'train/epoch': i,
                       'train/validation/dice_loss_val': valid_logs['dice_loss'],
                       'train/validation/iou_score_val': valid_logs['iou_score']})

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, os.path.join(
                    self.save_dir, f"model_{i}.pth"))

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')

            if i % plot_interval == 0:
                image, label = next(iter(val_loader))

                pr_mask = self.predict(image)
                pr_mask = (pr_mask.squeeze().cpu().numpy().round())

                self.visualize(
                    epoch=i,
                    mean=self.config['mean'],
                    std=self.config['std'],
                    num_classes=self.config['num_classes'],
                    image=image.squeeze().cpu().detach().numpy(),
                    ground_truth=label.squeeze().cpu().detach().numpy(),
                    prediction=pr_mask
                )

    def predict(self, data):
        return self.model.predict(data)
