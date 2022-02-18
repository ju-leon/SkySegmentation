# %%%
import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader
import numpy as np
from create_dataset import Dataset
from create_model import SegmentationModel


# %%

DATA_DIR = '/Users/leonjungemeyer/Files/StarGazerML/Converters/data/voc'

x_train_dir = os.path.join(DATA_DIR, 'JPEGImages')
y_train_dir = os.path.join(DATA_DIR, 'SegmentationClass')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

CLASSES = ['background', 'sky', 'clouds', 'foreground', 'subject']

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
# could be None for logits or 'softmax2d' for multiclass segmentation
ACTIVATION = 'sigmoid'
DEVICE = 'cpu'

# helper function for data visualization


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
    plt.show()
# %%
"""

augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    preprocessing_fn=smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS),
    classes=CLASSES,
)

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[10]
    visualize(image=image, mask=mask)
    print(mask.shape)

"""
# %%


model = SegmentationModel(ENCODER,
                          ENCODER_WEIGHTS,
                          CLASSES,
                          ACTIVATION,
                          DEVICE
                          )


# %%

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    preprocessing_fn=smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS),
    classes=CLASSES,
)

validation_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    preprocessing_fn=smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS),
    classes=CLASSES,
    augment=False
)



train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
valid_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

# %%

model.train(train_loader, valid_loader, 10)
# %%
