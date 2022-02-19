import os
from posixpath import split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

IMAGE_FORMAT = ".jpg"
LABEL_FORMAT = ".npy"


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        classes (list): All classes in the dataset. 0 should be background
        class_values (list): Classes that should be included in the dataset

    """

    def get_filename(self, string):
        return os.path.splitext(os.path.basename(string))[0]

    def __init__(
            self,
            images_dir,
            masks_dir,
            preprocessing_fn,
            num_classes,
            augment=True
    ):

        self.ids = list(map(self.get_filename, glob.glob(os.path.join(images_dir, '*' + IMAGE_FORMAT))))

        self.images_fps = [os.path.join(
            images_dir, image_id + IMAGE_FORMAT) for image_id in self.ids]
        self.masks_fps = [os.path.join(
            masks_dir, image_id + LABEL_FORMAT) for image_id in self.ids]

        self.class_values = list(range(num_classes))

        self.preprocessing_fn = preprocessing_fn

        if augment:
            self.augmentation = self.get_training_augmentation()
        else:
            self.augmentation = self.get_training_augmentation()

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.masks_fps[i])

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        # apply preprocessing
        sample = self.get_preprocessing()(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

    def get_training_augmentation(self):
        train_transform = [
            albu.augmentations.geometric.resize.LongestMaxSize(max_size=1000),

            albu.HorizontalFlip(p=0.5),

            albu.augmentations.geometric.transforms.ShiftScaleRotate(
                scale_limit=0.2, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.augmentations.transforms.PadIfNeeded(min_height=320, min_width=320,
                                                      always_apply=True, border_mode=0),
            albu.augmentations.crops.transforms.RandomCrop(
                height=320, width=320, always_apply=True),

            albu.augmentations.transforms.GaussNoise(p=0.2),
            albu.augmentations.geometric.transforms.Perspective(p=0.5),

            albu.OneOf(
                [
                    albu.augmentations.transforms.CLAHE(p=1),
                    albu.augmentations.transforms.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.augmentations.transforms.Sharpen(p=1),
                    albu.augmentations.transforms.Blur(blur_limit=3, p=1),
                    albu.augmentations.transforms.MotionBlur(
                        blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.augmentations.transforms.RandomBrightnessContrast(p=1),
                    albu.augmentations.transforms.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.augmentations.geometric.resize.LongestMaxSize(max_size=1000),

            albu.augmentations.transforms.PadIfNeeded(384, 480)
        ]
        return albu.Compose(test_transform)

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=self.preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)