import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot the training and validaton accuracy and loss curves

def plot_training_history(history, output_path):
    # Extract the training and validation accuracy
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy,'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(output_path)
    plt.show()


# Evaluate the model on the validation set
def evaluate_model(model, val_generator):
    loss , accuracy = model.evaluate(val_generator)
    return loss, accuracy

# Generate and plot the confusion matrix
def plot_confusion_matrix(model, val_generator, output_path):
    y_true = val_generator.classes
    y_pred = np.argmax(model.predict(val_generator), axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()
