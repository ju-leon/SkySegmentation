import torch
import numpy as np
import segmentation_models_pytorch as smp
import os

class SegmentationModel:

    def __init__(self, encoder, encoder_weights, classes, activation, device, save_dir) -> None:
        # create segmentation model with pretrained encoder
        self.model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
        )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder, encoder_weights)

        self.device = device
        self.save_dir = save_dir


    def train(self, train_loader, val_loader, epochs=10, lr=0.0001):
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
            
            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, os.path.join(self.save_dir, f"model_{i}.pth"))
                
            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
