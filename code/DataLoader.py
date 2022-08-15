import os
import numpy as np
from torchvision.datasets.cifar import CIFAR10

from ImageUtils import preprocess_image
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Optional, Tuple
from PIL import Image

"""This script implements the functions for reading data.
"""
def load_data(data_dir, preprocess_config):
    """Load the CIFAR-10 dataset.
    Args:
        data_dir: A string. The directory where data batches
            are stored.
        preprocess_config: Preprocess configurations stored in a file
    Returns:
        train_data: Data of type DataLoader Class
        test_data:  Data of type DataLoader Class
    """

    ### YOUR CODE HERE
    preprocess_train, preprocess_test = preprocess_image(preprocess_config)

    train_dataset = CIFAR10(root=data_dir,
                        download=True,
                        train=True,
                        transform=preprocess_train)

    test_dataset = CIFAR10(root=data_dir,
                        download=True,
                        train=False,
                        transform=preprocess_test)
    
    # Set 'pin_memory=True'. This lets 'DataLoader' allocate the samples in page-locked memory,
    # which speeds-up the transfer of data from CPU to GPU as we are usiing 'getitem' method.
    train_data = DataLoader(dataset=train_dataset,
                        batch_size=preprocess_config["batch_size"],
                        shuffle=True,
                        pin_memory=True,
                        num_workers=4)

    test_data = DataLoader(dataset=test_dataset,
                        batch_size=preprocess_config["batch_size"],
                        shuffle=False,
                        pin_memory=True,
                        num_workers=2)

    return train_data, test_data


def load_testing_images(data_dir, preprocess_config):
    """Load the images in private testing dataset.
    Args:
        data_dir: A string. The directory where the testing images
        are stored.
        preprocess_config: Preprocess configurations stored in a file

    Returns:
        private_data: Private data of type DataLoader Class
    """

    ### YOUR CODE HERE
    _, preprocess_test = preprocess_image(preprocess_config)

    private_dataset = Prediction_Data(root=data_dir,
                                transform=preprocess_test)
    
    private_data = DataLoader(dataset=private_dataset,
                            batch_size=preprocess_config["batch_size"],
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)

    return private_data
    
    ### END CODE HERE
    
    #return x_test


# Not using it as Dataset and Dataloader is used and we are gonna test it on private data 
def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

class Prediction_Data(Dataset):
    # Set private_data here.
    private_data = 'private_test_images_v3.npy'

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        data_path = os.path.join(root, self.private_data)
        self.data = np.load(data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.data[index]
        image=image.reshape(32,32,3)
        # conversion to Pyhton Image Library
        image = Image.fromarray(image)
        #image.save("abcd1234" + str(index) + ".png")
        if self.transform is not None:
            image = self.transform(image)

        return image