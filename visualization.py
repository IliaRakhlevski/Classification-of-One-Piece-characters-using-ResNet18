import torch
import matplotlib.pyplot as plt
import numpy as np
import math



# Plot training and validation loss curves over epochs
def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Plot training and validation accuracy curves over epochs
def plot_accuracy_curves(train_accuracies, val_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Plot confusion matrix to visualize classification performance across classes
def plot_confusion_matrix(cm, class_names): 

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_norm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            if value > 0.01:
                plt.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=10
                )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()
 


# Reverse normalization to display image correctly
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img
    
 
    
# Show misclassified images with predictions and confidence
def plot_misclassified(images, true_labels, pred_labels, confidences, class_names, cols=4):
    n = len(images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(16, 4 * rows))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)

        img = denormalize(images[i])
        img = img.permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.title(
            f"T: {class_names[true_labels[i]]}\n"
            f"P: {class_names[pred_labels[i]]}\n"
            f"{confidences[i]:.2f}",
            fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()
