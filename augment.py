import os
from dotenv import load_dotenv

load_dotenv()
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from genericpath import exists
import cv2


def main(image):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    datagen = ImageDataGenerator(
        .flow(img, batch_size=1,
              save_to_dir=augmented_sub_folder_path,
              # the augmented image is saved in  the augmented_sub_folder_path
              save_prefix='augmented', save_format=fileformat):


    if __name__ == "__main__":
       main(image)
