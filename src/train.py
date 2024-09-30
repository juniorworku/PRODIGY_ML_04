import os
from tensorflow.keras.callbacks import ModelCheckpoints
from data_preprocessing import load_data, get_class_names
from model import build_model
from utils import plot_training_history

# Parameters
data_dir = 'datasets/leapGestRecog'
img_size = (64, 64)
batch_size =32
epochs = 20

# Load data
train_generator, val_generator = load_data(data_dir, img_size, batch_size)
input_shape = (img_size[0], img_size[1], 3) # RGB images
num_classes = len(get_class_names(train_generator))

# Build model
model = build_model(input_shape, num_classes)

# Save the best model during training
checkpoints_path = 'models/best_model.h5'
checkpoint = ModelCheckpoints(checkpoints_path, mointor='val_accuracy', verbose=1, save_best_only=True, mode='max')

 # Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

# Save the training history plot
plot_training_history(history, output_path='results/training_history.png')