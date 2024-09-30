from tensorflow.keras.model import load_model
from data_preprocessing import load_data
from utils import plot_confusion_matrix, evaluate_model

 # Parameters
data_dir = 'datasets/leapGestRecog'
img_size = (64, 64)
batch_size = 32
model_path = 'models/best_model.h5'

# Load validation data
_, val_generator = load_data(data_dir, img_size, batch_size)

# Load the Saved model
model = load_model(model_path)

# Evalute the Model on the validation set
loss, accuracy = evaluate_model(model, val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Plot confusion matrix
plot_confusion_matrix(model, val_generator, output_path='results/confusion_matrix.png')