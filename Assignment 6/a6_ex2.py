import os
import glob
from typing import Union, Sequence
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

#implements two classes: ImageDataset and TransformedImageDataset -> extend the torch.utils.data.Dataset class
#these classes are designed to provide datasets of images and transformed images for machine learning or deep learning tasks.

class ImageDataset(Dataset):                   #extends the Dataset class
    def __init__(self, image_dir: str):        #takes an image_dir parameter(the directory where the images are located)
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)) #uses glob.glob to find all image file paths within the image_dir, with the extension .jpg.
        #The found file paths are stored in the image_paths attribute, sorted in ascending order

    def __getitem__(self, index: int):         #allows accessing individual images from the dataset based on their index.
        image_path = self.image_paths[index]   #takes an 'index' parameter, retrieves the corresponding image path from image_paths
        image = Image.open(image_path)         #uses Image.open to open the image as a pillow image
        return image, index                    #returns the image and its index

    def __len__(self):                         #returns the total number of images in the dataset
        return len(self.image_paths)           #by returning the length of the 'image_paths' list


class TransformedImageDataset(Dataset):       #extends the Dataset class
    def __init__(self, dataset: ImageDataset, image_size: Union[int, Sequence[int]]):
        self.dataset = dataset                #takes a 'dataset' parameter, which is an instance of the ImageDataset class defined earlier.
        self.image_size = image_size          #takes an image_size parameter, which represents the desired size of the transformed images
        #dataset and image_size values are stored as attributes

    def __getitem__(self, index: int):        #retrieves an individual image from the dataset based on the given index
        original_image, _ = self.dataset[index]       #access the original image from the ImageDataset
        transformed_image = random_augmented_image(original_image, self.image_size, index)   #applies a transformation('random_augmented_image'from ex.1) to original image using image_size and index as parameters
        return transformed_image, index   #returns the transformed image and its index

    def __len__(self):                     #returns the length of the dataset attribute,represents the number of transformed images in the datase
        return len(self.dataset)


if __name__ == "__main__":
    imgs = ImageDataset(image_dir="08_example_image.jpg")

    transformed_imgs = TransformedImageDataset(imgs, image_size=300)

    for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title("Original image")
        axes[1].imshow(transformed_img.permute(1, 2, 0))
        axes[1].set_title("Transformed image")
        fig.suptitle(f"Image {index}")
        plt.show()