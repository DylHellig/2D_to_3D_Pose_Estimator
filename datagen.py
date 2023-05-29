import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import tensorflow as tf
# pyright: reportMissingImports=false
from tensorflow.keras.utils import to_categorical
import models as my_models
import utils
import json
import matplotlib.pyplot as plt
import os
import cv2

def get_filenames(img_path, dataset):
    
    img_path = os.path.join(img_path, dataset)
    img_classes = os.listdir(img_path)
    
    filenames = []
    labels = []

    for class_name in img_classes:
        class_path = os.path.join(img_path, class_name)
        class_files = os.listdir(class_path)
        # filenames += class_files
        for img_file  in class_files:
            file_path = os.path.join(class_path, img_file)
            filenames.append(file_path)
            labels.append(int(class_name))
            
    return filenames, labels
            
def translate_image(image, trans_x, trans_y):
   rows, cols = image.shape[0], image.shape[1]
   T = np.float32([[1, 0, trans_x],
                 [0, 1, trans_y]])
   img_t = cv2.warpAffine(image, T, (cols, rows))
   img_t = img_t.reshape(rows, cols, 1)
   return img_t

def rotate_image(image, rotation_angle):
    # Get the image dimensions (height, width)
    height, width = image.shape[:2]

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)

    # Apply the rotation using warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    rotated_image = rotated_image.reshape(height, width, 1)
    return rotated_image

def translate_and_rotate(image, trans_x, trans_y, rotation_angle):
   
   image = translate_image(image, trans_x, trans_y)
   image = rotate_image(image, rotation_angle)
   
   return image

def height_data_generator(filenames, labels, batch_size):
    
   num_samples = len(filenames)
   indices = np.arange(num_samples)
   
   while True:
      np.random.shuffle(indices)
      
      for step in range(0, num_samples, batch_size):
        batch_indices = indices[step:step + batch_size]
        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            
            image_file = filenames[idx]
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            image = utils.format_image(image, (224, 224, 1))
            label = labels[idx]
            
            batch_images.append(image)
            batch_labels.append(label)
            
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        # batch_labels = to_categorical(batch_labels, 201)
        
        yield (batch_images, batch_labels)
            
def height_aug_data_generator(filenames, labels, batch_size):
    
   num_samples = len(filenames)
   indices = np.arange(num_samples)
   
   translations = np.arange(-30, 31, 5)
   rotations = np.arange(-180, 181, 10)
   
   while True:
      np.random.shuffle(indices)
      
      for step in range(0, num_samples, batch_size):
        batch_indices = indices[step:step + batch_size]
        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            
            image_file = filenames[idx]
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            image = utils.format_image(image, (224, 224, 1))
            translation_x = utils.random_sample(translations, 1)
            translation_y = utils.random_sample(translations, 1)
            rotation = utils.random_sample(rotations, 1)
            image = translate_and_rotate(image, translation_x, translation_y, rotation)
            
            label = labels[idx]
            
            batch_images.append(image)
            batch_labels.append(label)
            
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        # batch_labels = to_categorical(batch_labels, 201)
        
        yield (batch_images, batch_labels)
        
def generate_first_images(volume):
    
    center = np.array(volume.shape)//2
    
    Z = np.arange(center[2]-100, center[2]+100+1, 1)
    Y = center[1]*np.ones(len(Z))
    X = center[0]*np.ones(len(Z))
    points = np.column_stack([X, Y, Z])
    
    images1 = []
    for point in points:
        angle = np.array([0, 0, 0])
        img1 = utils.oblique_slice(point, angle, volume)
        img1 = utils.format_image(img1, (224, 224, 1))
        images1.append(img1)
        
    return np.array(images1)

def get_filenames_and_heights(img_dir, lbl_dir, dataset):
    
    img_path = os.path.join(img_dir, dataset)
    lbl_path = os.path.join(lbl_dir, dataset)
    
    img_classes = os.listdir(img_path)
    
    img_filenames = []
    lbl_filenames = []
    heights = []

    for class_name in img_classes:
        img_class_path = os.path.join(img_path, class_name)
        lbl_class_path = os.path.join(lbl_path, class_name)
        
        img_class_files = os.listdir(img_class_path)
        lbl_class_files = [ i.split('.')[0]+'.txt' for i in img_class_files ]
        # filenames += class_files
        for img_file, lbl_file  in zip(img_class_files, lbl_class_files):
            
            img_file_path = os.path.join(img_class_path, img_file)
            lbl_file_path = os.path.join(lbl_class_path, lbl_file)
            
            img_filenames.append(img_file_path)
            lbl_filenames.append(lbl_file_path)
            heights.append(int(class_name))
            
    return img_filenames, lbl_filenames, heights

def rot_data_generator(volume, first_images, img_filenames, lbl_filenames, heights, batch_size):
    
    center = np.array(volume.shape)//2
    
    num_samples = len(img_filenames)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        
        for step in range(0, num_samples, batch_size):
            batch_indices = indices[step:step + batch_size]
            
            batch_images1 = []
            batch_images2 = []
            batch_labels = []
            

            for idx in batch_indices:
                
                height = heights[idx]
                img1 = first_images[height]
                
                img2_file = img_filenames[idx]
                img2 = cv2.imread(img2_file, cv2.IMREAD_GRAYSCALE)
                img2 = utils.format_image(img2, (224, 224, 1))
                
                lbl_file = lbl_filenames[idx]
                with open(lbl_file, 'r') as f:
                    label = np.array(f.read().split(' '), dtype=float)
                
                batch_images1.append(img1)
                batch_images2.append(img2)
                batch_labels.append(label)
            
            batch_images1 = np.array(batch_images1)
            batch_images2 = np.array(batch_images2)
            batch_labels = np.array(batch_labels)
            
            yield ([batch_images1, batch_images2], batch_labels)

def rot_aug_data_generator(volume, first_images, img_filenames, lbl_filenames, heights, batch_size):
    
    center = np.array(volume.shape)//2
    
    num_samples = len(img_filenames)
    indices = np.arange(num_samples)
    
    translations = np.arange(-30, 31, 5)
    rotations = np.arange(-180, 181, 10)
    
    while True:
        np.random.shuffle(indices)
        
        for step in range(0, num_samples, batch_size):
            batch_indices = indices[step:step + batch_size]
            
            batch_images1 = []
            batch_images2 = []
            batch_labels = []
            

            for idx in batch_indices:
                
                height = heights[idx]
                img1 = first_images[height]
                
                img2_file = img_filenames[idx]
                img2 = cv2.imread(img2_file, cv2.IMREAD_GRAYSCALE)
                img2 = utils.format_image(img2, (224, 224, 1))
                
                translation_x = utils.random_sample(translations, 1)
                translation_y = utils.random_sample(translations, 1)
                rotation = utils.random_sample(rotations, 1)
                # aug_img1 = translate_and_rotate(img1, translation_x, translation_y, rotation)
                aug_img2 = translate_and_rotate(img2, translation_x, translation_y, rotation)
                
                lbl_file = lbl_filenames[idx]
                with open(lbl_file, 'r') as f:
                    label = np.array(f.read().split(' '), dtype=float)
                
                batch_images1.append(img1)
                batch_images2.append(img2)
                batch_labels.append(label)
                
                batch_images1.append(img1)
                batch_images2.append(aug_img2)
                batch_labels.append(label)
            
            batch_images1 = np.array(batch_images1)
            batch_images2 = np.array(batch_images2)
            batch_labels = np.array(batch_labels)
            
            yield ([batch_images1, batch_images2], batch_labels)