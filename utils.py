import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os 
import os.path as osp
import shutil
import cv2
import scipy.ndimage as ndimage
from scipy.spatial.transform import Rotation
from scipy import ndimage
import pandas as pd
import random
import concurrent.futures
import time

def plot_slices(fuse_img, slice_dir='z'):
   i = 0
   if slice_dir == 'y':
      i = 1
   if slice_dir == 'z':
      i = 2
   
   fig, ax = plt.subplots()
   plt.subplots_adjust(bottom=0.25)
   s = fuse_img.shape[i]//2
   
   if slice_dir == 'z':
      im = ax.imshow(fuse_img[s,:,:], cmap='gray')
   if slice_dir == 'y':
      im = ax.imshow(fuse_img[:,s,:], cmap='gray')
   if slice_dir == 'x':
      im = ax.imshow(fuse_img[:,:,s], cmap='gray')

   ax_slice = plt.axes([0.2, 0.1, 0.65, 0.03])
   slider = Slider(
      ax=ax_slice,
      label='Slice',
      valmin=0,
      valmax=fuse_img.shape[i]-1,
      valinit=s,
      valstep=1
   )

   def update(s):
      s = int(slider.val)
      if slice_dir == 'x':
         im.set_data(fuse_img[s,:,:])
         fig.canvas.draw_idle()
      if slice_dir == 'y':
         im.set_data(fuse_img[:,s,:])
         fig.canvas.draw_idle()
      if slice_dir == 'z':
         im.set_data(fuse_img[:,:,s])
         fig.canvas.draw_idle()
      
      
   slider.on_changed(update)
   # if slice_dir == 'x' or 'y':
   # ax.invert_yaxis()
   plt.show()
   
def euler_to_rot(rotations):
   euler_angles = np.deg2rad(rotations)
   R = Rotation.from_euler('XYZ', euler_angles).as_matrix()
   return R

def rot_to_euler(R):
   r = Rotation.from_matrix(R)
   euler_matrix = r.as_euler('XYZ', degrees=True)
   return euler_matrix

def euler_to_normal(rotations):
   euler_angles = np.deg2rad(rotations)
   R = Rotation.from_euler('XYZ', euler_angles).as_matrix()
   normal = R[:,2]
   return normal

def get_plane(point, normal, volume):
   xx, yy, = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), indexing='ij')
   a, b, c = normal
   x0, y0, z0 = point
   d = -a*x0 - b*y0 - c*z0
   zz = (-a*xx - b*yy - d)/c
   return zz

def oblique_slice(point, euler, volume):
   normal = euler_to_normal(euler)
   x_len, y_len, z_len = volume.shape
   #get plane using point and normal
   xx, yy, = np.meshgrid(np.arange(x_len), np.arange(y_len), indexing='ij')
   a, b, c = normal
   x0, y0, z0 = point
   d = -a*x0 - b*y0 - c*z0
   zz = (-a*xx - b*yy - d)/c
   #get coordinate array for plane
   plane_coords = np.vstack((xx.flatten(), yy.flatten(), zz.flatten()))
   #interpolate volume to plane coordinates
   slice_values = ndimage.map_coordinates(volume, plane_coords, order=1)
   #reshape to get slice images
   slice_img = slice_values.reshape(x_len, y_len)
   slice_img = ndimage.rotate(slice_img, euler[2], reshape=False)
   
   return slice_img
   

def generate_points(point, max_range):
   x0, y0, z0 = point
   x_max, y_max, z_max= [x0+max_range, y0+max_range, z0+max_range]
   x_grid, y_grid, z_grid = np.meshgrid(np.arange(x0, x_max), np.arange(y0, y_max), np.arange(z0, z_max))
   # z_grid = np.ones((max_range, max_range))*z0
   points = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
   return points 

def generate_angles(angle_range, incr):
   start, end = angle_range
   x_angles = np.arange(start, end+1, incr, dtype='float')
   y_angles = np.arange(start, end+1, incr, dtype='float')
   z_angles = np.arange(start, end+1, incr, dtype='float')
   x_grid, y_grid, z_grid = np.meshgrid(x_angles, y_angles, z_angles)
   angles = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
   return angles

def generate_slices(points_angles, volume, save_dir):
   
   for i, point_angle in enumerate(points_angles):
      point, rotations = point_angle
      normal = euler_to_normal(rotations)
      mri_slice = oblique_slice(point, normal, volume)
      mri_slice_normalized = cv2.normalize(mri_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
      filename = os.path.join(save_dir, str(i+1))
      print(filename)
      # cv2.imwrite(filename, mri_slice_normalized)

def generate_unique_pairs(start, end):
   pairs = [(i,j) for i in range(start, end+1) for j in range(i+1, end+1)]
   return pairs

def location_matrix(point, euler):
   R = euler_to_rot(euler)
   M = np.eye(4)
   M[:3,:3] = R
   M[:3, 3] = point 
   return M

def all_image_data(image_path, points_angles):
   #get sorted list of image filenames
   files = os.listdir(image_path)
   id_numbers = sorted([int(f.replace('.png', '')) for f in files])
   img_ids = [f'{str(num)}.png' for num in id_numbers]
   #store point and rotation data for each image id in dictionary
   image_data = {}
   for img_id, point_angle in zip(img_ids, points_angles):
      
      point, rotations = point_angle
      # data = point.tolist() + rotations.tolist()
      data = rotations.tolist()
      image_data[img_id] = data
      
   return image_data

# def generate_labels(image_path, label_path, points_angles):
   
#    slice_data = all_image_data(image_path, points_angles)
#    id_pairs = [(f'{str(pair[0])}.png', f'{str(pair[1])}.png') for pair in pairs]
#    transformations = {}
#    for i, pair in enumerate(id_pairs):
#       img1_data = np.array(slice_data[pair[0]])
#       img2_data = np.array(slice_data[pair[1]])
#       transf_data = img1_data - img2_data
#       transformations[str(i+1)] = transf_data.tolist()
      
#    for name, data in transformations.items():
#       filename = os.path.join(label_path, f'{name}.txt')
#       with open(filename, 'w') as f:
#          f.write(f'{str(data[0])} {str(data[1])} {str(data[2])}\n')
         
def train_test_split(all_ids, test_size):
   train_size = 1-test_size
   total_size = len(all_ids)
   train = random.sample(all_ids, int(train_size*total_size))
   test = [s for s in all_ids if s not in train]
   return train, test

def generate_pairs_sample(volume, initial_point=[197, 233, 189], voxel_range=100, angle_range=[-5, 5], trans_range= [-10, 10], num_pairs=10):
   
   points = generate_points(initial_point, voxel_range) #generate list of points
   angles = generate_angles(angle_range, 1) #generate list of rotations
   transformations = generate_angles(trans_range, 1) #generate list of transformations
   point = random.sample(points.tolist(), 1)[0] #randomly select one point
   rot_1 = random.sample(angles.tolist(), 1)[0] #randomly select one rotation
   img1 = oblique_slice(point, rot_1, volume) #generate initial slice
   
   rotations = random.sample(transformations.tolist(), num_pairs) #generate 10 random rotations for 2nd image in each pair
   
   images_1 = []
   images_1.append(img1)
   images_2 = []
   labels = []
   rot_1 = np.array(rot_1)
   for rot in rotations:
      rot = np.array(rot)
      rot_2 = rot_1 + rot
      img2 = oblique_slice(point, rot_2, volume)
      images_2.append(img2)
      labels.append(rot)
   
   return np.array(images_1), np.array(images_2), np.array(labels)

def data_generator(num_threads, function_args):
   
   args = [function_args]*num_threads
   with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
      results = list(executor.map(generate_pairs_sample, args))
   images_1 = np.concatenate([data[0] for data in results], axis=0)
   images_2 = np.concatenate([data[1] for data in results], axis=0)
   labels = np.concatenate([data[2] for data in results], axis=0)
   
   return images_1, images_2, labels

def save_image(filename, img):
   img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
   cv2.imwrite(filename, img_norm)

def tik():
   return time.perf_counter()

def tok(t):
   return time.perf_counter() - t 

def sort_key(name):
   num1, num2, _ = name.split('.')
   return int(num1), int(num2)

def write_data(img_path, label_path, images_1, images_2, labels):
   
   files = os.listdir(img_path)
   start_num = 0
   if len(files)>0:
      sorted_files = sorted(files, key=sort_key)
      start_num = int(sorted_files[-1].split('.')[0])
   
   for i, img1 in enumerate(images_1):
      file_num = i+1+start_num
      img1_filename = osp.join(img_path, f'{file_num}.0.png')
      save_image(img1_filename, img1)
      
      for j in range(0, 10):
         img2 = images_2[j+10*i]
         img2_filename = osp.join(img_path, f'{file_num}.{j+1}.png')
         save_image(img2_filename, img2)
         
         label = labels[j+10*i]
         label_filename = osp.join(label_path, f'{file_num}.{j+1}.txt')
         with open(label_filename, 'w') as f:
            f.write(f'{str(label[0])} {str(label[1])} {str(label[2])}')
            
def prep_image(img_path, input_shape):
   img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
   img = cv2.resize(img, (input_shape[0], input_shape[1]))
   img = img/255.0
   img = img.reshape(input_shape)
   return img

def format_image(img, input_shape):
   img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
   img = cv2.resize(img, (input_shape[0], input_shape[1]))
   img = img/255.0
   img = img.reshape(input_shape)
   return img

def data_generator(img_path, label_path):
   files = os.listdir(img_path)
   sorted_files = sorted(files, key=sort_key)
   num_sequences = int(sorted_files[-1].split('.')[0])
   first_images = []
   second_images = []
   transformations = []

   pair_list = []
   for i in range(1,num_sequences+1):
      id1 = f'{i}.0'
      ids_2 = []
      for j in range(1,11):
         id2 = f'{i}.{j}'
         ids_2.append(id2)
      pair_list.append([id1, ids_2])

   for pairs in pair_list:
      id1, ids2, = pairs
      img1_id = f'{id1}.png'
      img1_file = osp.join(img_path, img1_id)
      img1 = prep_image(img1_file, (224, 224, 1))
      for id2 in ids2:
         img2_id = f'{id2}.png'
         label_id = f'{id2}.txt'
         img2_file = osp.join(img_path, img2_id)
         img2 = prep_image(img2_file, (224, 224, 1))
         label_file = osp.join(label_path, label_id)
         with open(label_file, 'r') as f:
            rotations = np.array(f.read().split(' '), dtype=float)
         first_images.append(img1)
         second_images.append(img2)
         transformations.append(rotations)
   
   return np.array(first_images), np.array(second_images), np.array(transformations)

def collate_slices(us_path):
   
   slices = sorted(os.listdir(us_path))
   us_data = np.zeros((len(slices), 480, 640))
   for i, slc in enumerate(slices):
      us_file = osp.join(us_path, slc)
      slice_data = nib.load(us_file).get_fdata()[0]
      slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
      slice_data = slice_data/255.0
      us_data[i,:,:] = slice_data
   us_data = us_data.transpose(1, 2, 0)
   
   return us_data

def random_sample(np_array, sample_size):
   sample = np.array(random.sample(np_array.tolist(), sample_size))
   if(sample_size==1):
      return sample[0]
   else:
      return sample
   
def load_volume(path):
   data = np.array(nib.load(path).get_fdata())
   return data

def generate_shifts(max_shift):
   x_grid, y_grid, z_grid = np.meshgrid(np.arange(max_shift+1, dtype='float'), np.arange(max_shift+1, dtype='float'), np.arange(max_shift+1, dtype='float'))
   shifts = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
   return shifts

def generate_heights(volume, range_from_center):
   center = np.array(volume.shape)//2
   Z = np.arange(center[2]-range_from_center, center[2]+range_from_center+1, 1)
   Y = center[1]*np.ones(len(Z))
   X = center[0]*np.ones(len(Z))
   points = np.column_stack([X, Y, Z])
   
   return points

def generate_angles_XY(angle_range):
   angle_range = np.arange(angle_range[0], angle_range[1]+1, 1)
   x_grid, y_grid = np.meshgrid(angle_range, angle_range)
   z_grid = np.zeros((len(angle_range), len(angle_range)))
   angles = np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
   
   return angles

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

def find_rotation(img1, img2):
   
   sift = cv2.SIFT_create()
   
   k1, d1 = sift.detectAndCompute(img1, None)
   k2, d2 = sift.detectAndCompute(img2, None)
   
   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
   
   matches = bf.match(d1, d2)
   
   matches = sorted(matches, key=lambda x: x.distance)

   # Extract keypoints for matched points
   src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
   dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

   # Compute affine transformation (RANSAC-based)
   M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

   # Extract translation from the affine transformation
   translation = M[:, 2]
   rotation_matrix = M[:, :2]
   rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

   return rotation_angle

def find_translation(img1, img2):
   
   sift = cv2.SIFT_create()
   
   k1, d1 = sift.detectAndCompute(img1, None)
   k2, d2 = sift.detectAndCompute(img2, None)
   
   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
   
   matches = bf.match(d1, d2)
   
   matches = sorted(matches, key=lambda x: x.distance)

   # Extract keypoints for matched points
   src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
   dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

   # Compute affine transformation (RANSAC-based)
   M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

   # Extract translation from the affine transformation
   translation = M[:, 2]
   rotation_matrix = M[:, :2]
   rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

   return translation

def find_translation_and_rotation(img1, img2):
   rotation = find_rotation(img1, img2)
   img2 = rotate_image(img2, rotation)
   translation = find_translation(img1, img2)
   
   return translation[0], translation[1], -rotation