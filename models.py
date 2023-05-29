import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# pyright: reportMissingImports=false
from tensorflow.keras import layers, Model, models, Input

def cnn(input_shape):
   inputs = tf.keras.Input(shape=input_shape)
   x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Conv2D(64, (3, 3), activation='relu')(x)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Conv2D(128, (3, 3), activation='relu')(x)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Flatten()(x)
   x = layers.Dense(256, activation='relu')(x)
   outputs = layers.Dense(128, activation='linear')(x)
   model = models.Model(inputs=inputs, outputs=outputs)
   return model


def relative_rotation_network(input_shape):
   base_cnn = cnn(input_shape)
   input_a = tf.keras.Input(shape=input_shape)
   input_b = tf.keras.Input(shape=input_shape)
   processed_a = base_cnn(input_a)
   processed_b = base_cnn(input_b)
   features1 = layers.Flatten()(processed_a)
   features2 = layers.Flatten()(processed_b)
   distance = layers.Subtract()([features1, features2])
   rotation = layers.Dense(128, activation='relu')(distance)
   # rotation = layers.Dropout(0.5)(rotation)
   rotation = layers.Dense(2, activation='linear', name="rotation")(rotation)

   return tf.keras.Model(inputs=[input_a, input_b], outputs=rotation)

def height_estimation_network(input_shape):
   inputs = tf.keras.Input(shape=input_shape)
   x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Conv2D(64, (3, 3), activation='relu')(x)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Conv2D(128, (3, 3), activation='relu')(x)
   x = layers.MaxPooling2D((2, 2))(x)
   x = layers.Flatten()(x)
   x = layers.Dense(256, activation='relu')(x)
   outputs = layers.Dense(1, activation='linear')(x)
   model = models.Model(inputs=inputs, outputs=outputs)
   return model

# def height_estimation_network(input_shape, num_classes):
#    inputs = Input(shape=input_shape)
#    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
#    x = layers.MaxPooling2D((2, 2))(x)
#    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
#    x = layers.MaxPooling2D((2, 2))(x)
#    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
#    x = layers.Flatten()(x)
#    x = layers.Dense(64, activation='relu')(x)
#    outputs = layers.Dense(num_classes, activation='softmax')(x)
#    model = models.Model(inputs=inputs, outputs=outputs)
#    return model