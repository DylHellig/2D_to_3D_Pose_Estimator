import models as my_models
import tensorflow as tf

input_shape = (224, 224, 1)
num_classes = 201

def rotation_estimation_model(weight_file):
   rotation_model = my_models.relative_rotation_network(input_shape=input_shape)
   rotation_model.load_weights(weight_file)
   rotation_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
   
   return rotation_model

def height_estimation_model(weight_file):
   height_model = my_models.height_estimation_network(input_shape=input_shape)
   height_model.load_weights(weight_file)
   height_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae'])
   
   return height_model