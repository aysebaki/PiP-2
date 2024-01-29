import random
from typing import Union, Sequence
import torch
from PIL import Image
import torchvision.transforms as transforms

#takes input image -> applies random chain of transformations -> returns the transformed image as a PyTorch tensor
#customized image size, ensures reproducibility by using a seed, and follows a predefined chain of transformations
#goal:to create a flexible and reusable function for generating augmented images for tasks such as image classification or object detection

def random_augmented_image(image: Image.Image,                      #input image loaded with pillow
                           image_size: Union[int, Sequence[int]],   #desired image size
                           seed: int                                #seed for reproducibility
                           ) -> torch.Tensor:

    random.seed(seed)     #ensuring the random transformations applied will be reproducible

    resize_transform = transforms.Resize(image_size)    #created using the 'image_size' parameter,will resize image to specified size

#The block defines a list of available transformation classes

    available_transforms = [
        transforms.RandomRotation,
        transforms.RandomVerticalFlip,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter
    ]
    selected_transforms = random.sample(available_transforms, 2) #randomly selects two transformations from the list

#the original image is assigned to transformed_image
#Then the selected transformations are applied one by one to the transformed_image using the instantiated transformation objects.

    transformed_image = image
    for transform in selected_transforms:
        transformed_image = transform()(transformed_image)

    #create two transformation objects
    to_tensor_transform = transforms.ToTensor()   #converts the image to a PyTorch tensor
    dropout_transform = torch.nn.Dropout()        #applies dropout regularization to the tensor

    transformed_image = resize_transform(transformed_image)       #resize transformation
    transformed_image = to_tensor_transform(transformed_image)    #conversion to a PyTorch tensor
    transformed_image = dropout_transform(transformed_image)      #dropout regularization

    return transformed_image      #transformed image is returned as a PyTorch tensor

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    with Image.open("08_example_image.jpg") as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show()