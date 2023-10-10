import os
import glob
import pandas as pd
import numpy as np

class ImageDataPreparation:
    def __init__(self, main_path):
        self.main_path = main_path
        self.train_path = os.path.join(main_path, "train")
        self.test_path = os.path.join(main_path, "test")
        self.val_path = os.path.join(main_path, "val")

    def load_data(self, folder):
        normal_images = glob.glob(os.path.join(self.main_path, folder, "NORMAL", "*.jpeg"))
        pneumonia_images = glob.glob(os.path.join(self.main_path, folder, "PNEUMONIA", "*.jpeg"))

        image_list = normal_images + pneumonia_images
        labels = ['Normal'] * len(normal_images) + ['Pneumonia'] * len(pneumonia_images)

        return pd.DataFrame({'image': image_list, 'class': labels})

    def prepare_data(self):
        df_train = self.load_data("train")
        df_test = self.load_data("test")
        df_val = self.load_data("val")

        return df_train, df_test, df_val

# Exemple d'utilisation de la classe
data_preparator = ImageDataPreparation("Dataset/chest_xray")
df_train, df_test, df_val = data_preparator.prepare_data()
