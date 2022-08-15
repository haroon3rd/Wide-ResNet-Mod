import torch
import random
import numpy as np
from torchvision import transforms
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps

def parse_record(images): 
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    #preprocess_train, preprocess_test = preprocess_image(preprocess_config)
    # Reshape from [depth * height * width] to [depth, height, width].
    images = images.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    images = np.transpose(images, [1, 2, 0])

    #images = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    images = np.transpose(images, [2, 0, 1])

    ### END CODE HERE

    #image = preprocess_image(image, training) # If any.
   
    return images 


def preprocess_image(preprocess_config):
    """
    Get the preprocessing transforms for training and testing images.
    The function normalizes, crops and flips the training images.
    Additionally, the cutout data augmentation has been added. This 
    will cut holes in the images so that the model can learn other
    salient aspects of the images.

    Args:
        model_config: Configuration dict containing the options for
        image preprocessing. Must contain the following options:
        1. crop : True/False
        2. crop_padding : Pixel padding for image cropping
        3. flip : True/False
        4. cutout : True/False
        5. cutout_holes : number of holes in the image
        6. cutout_length: length of the holes in the image

    Returns:
        preprocess_train: transformation for the training images
        preprocess_test: transformation for the testing images

    """
    ### YOUR CODE HERE

    # ref : https://github.com/kuangliu/pytorch-cifar/issues/19
    # for mean and sd values

    preprocess_train = transforms.Compose([])
    
    if preprocess_config['crop']:
        preprocess_train.transforms.append(transforms.RandomCrop(32, padding=preprocess_config['crop_padding']))
    if preprocess_config['flip']:
        preprocess_train.transforms.append(transforms.RandomHorizontalFlip())
    
    preprocess_train.transforms.append(AutoAugment())
    preprocess_train.transforms.append(transforms.ToTensor())
    preprocess_train.transforms.append(transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]]))

    if preprocess_config['cutout']:
        cutout_holes = preprocess_config["cutout_holes"]
        cutout_length = preprocess_config["cutout_length"]
        preprocess_train.transforms.append(Cutout(n_holes=cutout_holes, length=cutout_length))

    preprocess_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]) 

    return preprocess_train, preprocess_test


class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
            ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
            ['Color', 0.4, 3, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
            ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
            ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
            ['Brightness', 0.9, 6, 'Color', 0.2, 8],
            ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
            ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Color', 0.9, 9, 'Equalize', 0.6, 6],
            ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
            ['Brightness', 0.1, 3, 'Color', 0.7, 0],
            ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
            ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
            ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
            ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
        ]

    def __call__(self, image):
        image = apply_policy(image, self.policies[random.randrange(len(self.policies))])
        return image


operations = {
    'ShearX': lambda image, magnitude: shear_x(image, magnitude),
    'ShearY': lambda image, magnitude: shear_y(image, magnitude),
    'TranslateX': lambda image, magnitude: translate_x(image, magnitude),
    'TranslateY': lambda image, magnitude: translate_y(image, magnitude),
    'Rotate': lambda image, magnitude: rotate(image, magnitude),
    'AutoContrast': lambda image, magnitude: auto_contrast(image, magnitude),
    'Invert': lambda image, magnitude: invert(image, magnitude),
    'Equalize': lambda image, magnitude: equalize(image, magnitude),
    'Solarize': lambda image, magnitude: solarize(image, magnitude),
    'Posterize': lambda image, magnitude: posterize(image, magnitude),
    'Contrast': lambda image, magnitude: contrast(image, magnitude),
    'Color': lambda image, magnitude: color(image, magnitude),
    'Brightness': lambda image, magnitude: brightness(image, magnitude),
    'Sharpness': lambda image, magnitude: sharpness(image, magnitude),
    'Cutout': lambda image, magnitude: cutout(image, magnitude),
}


def apply_policy(image, policy):
    if random.random() < policy[1]:
        image = operations[policy[0]](image, policy[2])
    if random.random() < policy[4]:
        image = operations[policy[3]](image, policy[5])

    return image


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(image, magnitude):
    image = np.array(image)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
    image = Image.fromarray(image)
    return image


def shear_y(image, magnitude):
    image = np.array(image)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
    image = Image.fromarray(image)
    return image


def translate_x(image, magnitude):
    image = np.array(image)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, image.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
    image = Image.fromarray(image)
    return image


def translate_y(image, magnitude):
    image = np.array(image)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, image.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
    image = Image.fromarray(image)
    return image


def rotate(image, magnitude):
    image = np.array(image)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, image.shape[0], image.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    image = np.stack([ndimage.interpolation.affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset) for c in range(image.shape[2])], axis=2)
    image = Image.fromarray(image)
    return image


def auto_contrast(image, magnitude):
    image = ImageOps.autocontrast(image)
    return image


def invert(image, magnitude):
    image = ImageOps.invert(image)
    return image


def equalize(image, magnitude):
    image = ImageOps.equalize(image)
    return image


def solarize(image, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    image = ImageOps.solarize(image, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return image


def posterize(image, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    image = ImageOps.posterize(image, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    return image


def contrast(image, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    image = ImageEnhance.Contrast(image).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return image


def color(image, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    image = ImageEnhance.Color(image).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return image


def brightness(image, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return image


def sharpness(image, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return image


def cutout(org_image, magnitude=None):
    image = np.array(org_image)

    magnitudes = np.linspace(0, 60/331, 11)

    image = np.copy(org_image)
    mask_val = image.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(image.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    top = np.random.randint(0 - mask_size//2, image.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, image.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_val)

    image = Image.fromarray(image)

    return image

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    # Image with n_holes of dimension length x length cut out of it.
    def __call__(self, image):
        mask_val = np.ones((image.size(1), image.size(2)), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(image.size(1))
            x = np.random.randint(image.size(2))

            y1 = np.clip(y - self.length // 2, 0, image.size(1))
            y2 = np.clip(y + self.length // 2, 0, image.size(1))
            x1 = np.clip(x - self.length // 2, 0, image.size(2))
            x2 = np.clip(x + self.length // 2, 0, image.size(2))

            mask_val[y1: y2, x1: x2] = 0.

        mask_val = torch.from_numpy(mask_val)
        mask_val = mask_val.expand_as(image)
        image = image * mask_val

        return image