import os
import random
import matplotlib.pyplot as plt

def display_random_images(train_folder, num_ima=5):
    classes = os.listdir(train_folder)
    num_classes = len(classes)
    
    fig, axes = plt.subplots(num_classes, num_ima, figsize=(15, 5*num_classes))
    
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(train_folder, class_name)
        image_files = os.listdir(class_folder)
        
        random_images = random.sample(image_files, num_ima)
        
        for j, random_image in enumerate(random_images):
            image_path = os.path.join(class_folder, random_image)
            image = plt.imread(image_path)
            axes[i, j].imshow(image)
            axes[i, j].set_title(class_name)
            axes[i, j].axis('off')
    
    plt.show()

train_folder = 'Dataset/chest_xray/train'
display_random_images(train_folder)
