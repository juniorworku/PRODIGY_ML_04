import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(64,64), batch_size=32):
    # Load data from directory
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,     # Normalize images
        shear_range=0.2,       # Apply shearing transformation
        zoom_range=0.2,        # Apply zoom transformation
        horizontal_flip=True,  # Flip images horizontally
        validation_split=0.2   # Split data into training and validation sets
    )
    
    # Load training data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def get_class_names(train_generator):
    # Get class names
    return list(train_generator.class_indices.keys())