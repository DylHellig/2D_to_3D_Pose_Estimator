import numpy as np 
import tensorflow as tf
import models as my_models
import model_loader
import utils
import json
import matplotlib.pyplot as plt
import os
import cv2
import datagen
from sklearn.metrics import r2_score
import pandas as pd

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def get_test_data(img_path, lbl_path):
   
   img_path = './dataset5/images/'
   lbl_path = './dataset5/labels'
   test_filenames, test_rot_filenames, labels = datagen.get_filenames_and_heights(img_path, lbl_path, 'test')
   
   test_images = []
   test_rots = []
   test_labels = []
   test_aug_images = []
   test_aug_rots = []
   test_aug_labels = []
   test_augmentations = []
   translations = np.arange(-30, 31, 5)
   rotations = np.arange(-180, 181, 10)
   
   for img_file, rot_file, label in zip(test_filenames, test_rot_filenames, labels):
      img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
      img = utils.format_image(img, (224, 224, 1))
      
      
      translation_x = utils.random_sample(translations, 1)
      translation_y = utils.random_sample(translations, 1)
      rotation = utils.random_sample(rotations, 1)
      img_aug = utils.translate_and_rotate(img, translation_x, translation_y, rotation)
      
      with open(rot_file, 'r') as f:
         rot = np.array(f.read().split(' '), dtype=np.float32)
      
      test_images.append(img)
      test_rots.append(rot)
      test_labels.append(label)
      
      test_aug_images.append(img)
      test_aug_rots.append(rot)
      test_aug_labels.append(label)
      test_augmentations.append(np.array([0,0,0]))
      test_aug_images.append(img_aug)
      test_aug_rots.append(rot)
      test_aug_labels.append(label)
      test_augmentations.append(np.array([translation_x, translation_y, rotation]))
      
   test_images = np.array(test_images)
   test_rots = np.array(test_rots)
   test_labels = np.array(test_labels, dtype=np.float32)
   test_aug_images = np.array(test_aug_images)
   test_aug_rots = np.array(test_aug_rots)
   test_aug_labels = np.array(test_aug_labels, dtype=np.float32)
   test_augmentations = np.array(test_augmentations)
   
   return (test_images, test_rots, test_labels), (test_aug_images, test_aug_rots, test_aug_labels, test_augmentations)

def evaluatuion_metrics(model, name):
   
   mse, rmse, mae = model.evaluate(test_images, test_heights)
   mse_aug, rmse_aug, mae_aug = model.evaluate(test_aug_images, test_aug_heights)
   
   eval_metrics = []
   eval_metrics.append((f'{name}', mse, np.sqrt(mse), mae, 0.5*mse, 0.5*np.sqrt(mse), 0.5*mae))
   eval_metrics.append((f'{name} with Augmentation', mse_aug, np.sqrt(mse_aug), mae_aug, 0.5*mse_aug, np.sqrt(0.5*mse_aug), 0.5*mae_aug))
   headers = ['Dataset', 'MSE_{voxels}', 'RMSE_{voxels}', 'MAE_{voxels}', 'MSE_{mm}', 'RMSE_{mm}', 'MAE_{mm}']
   df = pd.DataFrame(eval_metrics, columns=headers, index=None)
   
   return df

def height_predictions(model, test_images, true_heights, test_aug_images, true_aug_heights):
   pred_heights = model.predict(test_images).flatten()
   pred_aug_heights = model.predict(test_aug_images).flatten()
   
   return true_heights, pred_heights, true_aug_heights, pred_aug_heights

def plot_error_distribution(true1, preds1, true2, preds2):
   errors1 = 0.5*true1 - 0.5*preds1
   errors2 = 0.5*true2 - 0.5*preds2
   plt.figure(figsize=(12, 5))

   # Histogram
   plt.subplot(1, 2, 1)
   plt.hist(errors1, bins=50, alpha=0.8, label=("No Augmentation"), color='m')
   plt.hist(errors2, bins=50, alpha=0.3, label=("With Augmentation"), color='g')
   plt.xlabel('Error')
   plt.ylabel('Frequency')
   plt.title('Error Distribution (Histogram)')
   plt.legend()

   # Box plot
   plt.subplot(1, 2, 2)
   plt.boxplot([errors1, errors2], labels=["No Augmentation", "With Augmentation"])
   plt.ylabel('Error')
   plt.title('Error Distribution (Box plot)')

   plt.show()

def compare_images(img1, img2):
   # mse = mean_squared_error(img1, img2)
   ss = ssim(img1, img2, data_range=np.max(img1)-np.min(img1), channel_axis=2)
   # si = ssim(img1, img2, win_size=7, channel_axis=None)
   img1 = img1.flatten()
   img2 = img2.flatten()
   corr, _ = pearsonr(img1, img2)
   
   return corr, ss

def generate_predicted_images_height(pred_heights, true_rots, volume):
   pred_images = []
   for z, rot in zip(pred_heights, true_rots):
      r_x, r_y = rot
      point = np.array([197, 233, z+89], dtype=np.float32)
      angle = np.array([r_x, r_y, 0])
      pred_img = utils.oblique_slice(point, angle, volume)
      pred_img = utils.format_image(pred_img, (224, 224, 1))
      pred_images.append(pred_img)
      
   return np.array(pred_images)

def generate_predicted_images_rot(pred_rots, true_heights, volume):
   pred_images = []
   for rot, z in zip(pred_rots, true_heights):
      r_x, r_y = rot
      point = np.array([197, 233, z+89], dtype=np.float32)
      angle = np.array([r_x, r_y, 0])
      pred_img = utils.oblique_slice(point, angle, volume)
      pred_img = utils.format_image(pred_img, (224, 224, 1))
      pred_images.append(pred_img)
      
   return np.array(pred_images)

def generate_predicted_images_3dof(pred_pose3, volume):
   pred_images = []
   for pred in pred_pose3:
      z, r_x, r_y = pred
      point = np.array([197, 233, z+89], dtype=np.float32)
      angle = np.array([r_x, r_y, 0])
      pred_img = utils.oblique_slice(point, angle, volume)
      pred_img = utils.format_image(pred_img, (224, 224, 1))
      pred_images.append(pred_img)
      
   return np.array(pred_images)

def generate_predicted_images_inplane(pred_trans, test_images):
   pred_images = []
   for pred, img in zip(pred_trans, test_images):
      x, y, r_z = pred
      pred_img = utils.translate_and_rotate(img, x, y, r_z)
      pred_images.append(pred_img)
      
   return np.array(pred_images)

def generate_predicted_images_6dof(pred_pose6, volume):
   pred_images_aug = []
   for pred in pred_pose6:
      x, y, z, r_x, r_y, r_z = pred
      point = np.array([197, 233, z+89], dtype=np.float32)
      angle = np.array([r_x, r_y, 0])
      pred_img = utils.oblique_slice(point, angle, volume)
      pred_img = utils.format_image(pred_img, (224, 224, 1))
      pred_img = utils.translate_and_rotate(pred_img, x, y, r_z)
      pred_images_aug.append(pred_img)
      
   return np.array(pred_images_aug)

if __name__=='__main__':
   
   img_path = ''
   lbl_path = ''
   weight_path = ''
   name = ''
   
   model = model_loader.height_estimation_model('./weights/5deg_h_aug.h5')
   
   no_aug, aug = get_test_data(img_path, lbl_path)
   
   test_images, test_rots, test_heights = no_aug
   test_aug_images, test_aug_rots, test_aug_heights, test_augmentations = aug
   
   df = evaluatuion_metrics(model, name)
   
   true_heights, pred_heights, true_aug_heights, pred_aug_heights = height_predictions(model, test_images, test_heights, test_aug_images, test_aug_heights)
   
   
   
    
   