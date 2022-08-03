import os
from dotenv import load_dotenv

load_dotenv()
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from genericpath import exists
import cv2


# This file is used to augment the data images in test and train folder of the respective dataset.
# define data preparation
class image_generation:

    def data_augmentation(dir_image, augment_image, raw_file_path, enhanced_file_path, fileformat):

        # change the width and height by 1% for augmented images. The horizontal flip and vertical flip is set to false.
        data_gen = ImageDataGenerator(width_shift_range=0.01, height_shift_range=0.01, horizontal_flip=False,
                                      vertical_flip=False)
        for image_dir, augment_dir in zip(dir_image,
                                          augment_image):  # dir image= original folder, augment_image= augmented folder
            images_sub_folder_path = '{}/{}'.format(raw_file_path,
                                                    image_dir)  # images sub folder is the subfolders inside the original folder,
            # augmented sub folder is the subfolders inside the augmented folder
            augmented_sub_folder_path = '{}/{}'.format(
                enhanced_file_path, augment_dir)
            file_list = sorted(
                os.listdir(images_sub_folder_path))  # this is the sorted list of total amount of images per subfolder
            augmented_file_list = len(sorted(os.listdir(augmented_sub_folder_path)))
            if augmented_file_list == 0:
                for count, file_name in enumerate(file_list):
                    if exists(os.path.join(augmented_sub_folder_path,
                                           f'{file_name}')):  # if the augmented images already exists, delete them
                        os.remove(os.path.join(augmented_sub_folder_path, f'{file_name}'))
                    images_filepath = "{}\{}".format(images_sub_folder_path,
                                                     file_name)  # this is the original image path
                    img = cv2.imread(images_filepath)  # read the original image
                    img = img.reshape((
                                      1,) + img.shape)  # to augment data, you need to change the shape of the image, the 1 here represent 1 image
                    i = 0
                    # The below code is used to generater images and then save in the "augmented_sub_folder". The i value determines the number of images to be saved.
                    # In the case i =4, 5 augmented images are saved per original image
                    for batch in data_gen.flow(img, batch_size=1,
                                               save_to_dir=augmented_sub_folder_path,
                                               # the augmented image is saved in  the augmented_sub_folder_path
                                               save_prefix='augmented', save_format=fileformat):
                        i += 1
                        if i > 4:
                            break


# you can choose to run this file here or at the main.py as well. It's up to you to decide.
if __name__ == '__main__':
    database = ["hkpu",
                "fuvsm"]  # "utfvp","vera","plusvein"] #list all the database that you want to use for the project
    for x in database:
        for folder in ["train", "test"]:
            original_dir = os.getenv("{}_{}_file_structure".format(x, folder))
            augmented_dir = os.getenv("{}_{}_augment_file_structure".format(x, folder))
            original_folder_list = sorted(os.listdir(original_dir))
            augmented_folder_list = sorted(os.listdir(augmented_dir))
            image_generation.data_augmentation(original_folder_list, augmented_folder_list, original_dir, augmented_dir,
                                               'jpg')

