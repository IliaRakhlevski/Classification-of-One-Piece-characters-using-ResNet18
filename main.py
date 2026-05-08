import os
import numpy as np
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from visualization import plot_confusion_matrix, plot_loss_curves, plot_accuracy_curves, plot_misclassified
from utils import get_top_confident_mistakes, set_seed



TRAIN_DIR = 'Train'
VALID_DIR = 'Val'
TEST_DIR = 'Test'


BLACK = [0,0,0]
PAD_COLOR = BLACK   # color of the added pads
IMG_HEIGHT = int(224) 
IMG_WIDTH  = int(224)

CHANNELS = 3


CLASS_NAMES = [
    "Ace",
    "Akainu",
    "Brook",
    "Chopper",
    "Crocodile",
    "Franky",
    "Jinbei",
    "Kurohige",
    "Law",
    "Luffy",
    "Mihawk",
    "Nami",
    "Rayleigh",
    "Robin",
    "Sanji",
    "Shanks",
    "Usopp",
    "Zoro",
]

NUM_CLASSES = len(CLASS_NAMES)

class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}




# Add pads to image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def image_padding(image, img_height, img_width):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    top = 0
    bottom = 0
    left = 0
    right = 0
    
    if im_cur_height < img_height:
        deltha = (int)((img_height - im_cur_height) // 2)
        top = deltha
        bottom = deltha
        if (top + bottom + im_cur_height) < img_height:
            top += 1
        
    if im_cur_width < img_width:
        deltha = (int)((img_width - im_cur_width) // 2)
        left = deltha
        right = deltha
        if (left + right + im_cur_width) < img_width:
            left += 1
    
    pad_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    
    return pad_image



# Change the image in order it to have the given size (img_width, img_height)
# image - image (frame) object
# img_width - final image width
# img_height - final image height
def image_resizing(image, img_height, img_width):
    im_cur_height = image.shape[0]
    im_cur_width = image.shape[1]
    
    height_factor = im_cur_height / img_height
    width_factor  = im_cur_width  / img_width
    
    factor = width_factor
    if height_factor > width_factor:
        factor = height_factor
        
    im_cur_height = int(im_cur_height / factor)
    im_cur_width  = int(im_cur_width  / factor)
    
    image = cv2.resize(image, (im_cur_width, im_cur_height))
    
    image = image_padding(image, img_height, img_width)
    
    return image


# read image from the given file
# file_path - path to the specific file
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: failed to read {file_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # for PyTorch using
    res_img = image_resizing(img, IMG_HEIGHT, IMG_WIDTH)
    return res_img

 
# load data from the images list
# images - list of the images
def load_data_from_images(images):
    imgs_len = len(images)
    X = np.zeros((imgs_len, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    
    for i, image_file in enumerate(images):
        img = read_image(image_file)
        if img is not None:
            X[i,:] = img
        
    return X


# load data from the specific directory
# data_dir - directory containing images files
def load_data(data_dir):
    X = []
    y = []
    files_list = sorted(os.listdir(data_dir))
    for file_name in files_list:
        data_path = data_dir + "/" + file_name
        if os.path.isdir(data_path):
            images = [data_path + "/" + f for f in os.listdir(data_path)]
            images_data = load_data_from_images(images)
            
            X.extend(images_data)
            y.extend([class_to_idx[file_name]] * len(images_data))
            
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# convert images data to PyTorch format
# X - images data
# y - labels
def convert_to_torch(X, y):
    X_torch = torch.from_numpy(X).permute(0, 3, 1, 2).float()
    y_torch = torch.from_numpy(y).long()
    return X_torch, y_torch




#=================  Model  ==============================================

BATCH_SIZE = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create model
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    return model



if __name__ == '__main__':
    
    #set_seed(42)
    #set_seed(123)
    set_seed(777)
    
    train_set_x, train_set_y = load_data(TRAIN_DIR)
    valid_set_x, valid_set_y = load_data(VALID_DIR)
    test_set_x, test_set_y = load_data(TEST_DIR)
    
    train_set_x = train_set_x.astype("float32") / 255.0
    test_set_x = test_set_x.astype("float32") / 255.0
    valid_set_x = valid_set_x.astype("float32") / 255.0
    
    print(train_set_x.shape, train_set_y.shape)
    print(valid_set_x.shape, valid_set_y.shape)
    print(test_set_x.shape, test_set_y.shape)
    print(train_set_x.min(), train_set_x.max(), train_set_x.dtype)
    print(len(CLASS_NAMES), len(np.unique(train_set_y)))
    
    #============= Py Torch =================================
    
    train_x_torch, train_y_torch = convert_to_torch(train_set_x, train_set_y)
    valid_x_torch, valid_y_torch = convert_to_torch(valid_set_x, valid_set_y)
    test_x_torch, test_y_torch = convert_to_torch(test_set_x, test_set_y)
    
    # Normalize images using ImageNet statistics (required for pretrained ResNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    train_x_torch = (train_x_torch - mean) / std
    valid_x_torch = (valid_x_torch - mean) / std
    test_x_torch = (test_x_torch - mean) / std
        
    print(train_x_torch.dtype, train_y_torch.dtype)
    
    # Create PyTorch datasets from tensors
    train_dataset = TensorDataset(train_x_torch, train_y_torch)
    val_dataset = TensorDataset(valid_x_torch, valid_y_torch)
    test_dataset = TensorDataset(test_x_torch, test_y_torch)
    
    # Create data loaders for batching
    # Shuffle is enabled only for training to improve generalization
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    model = create_model()
    print(model.fc)
    print(next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    
    EPOCHS = 80

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []


    for epoch in range(EPOCHS):
        
        # ===== Train =====
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
    
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
    
            optimizer.zero_grad()
    
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    
        train_loss = running_loss / total
        train_acc = correct / total
    
        # ===== Validation =====
        
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
    
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
    
                val_loss_sum += loss.item() * x_batch.size(0)
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
    
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
    
    print(f'\nBest val: {best_val_acc}')
    
    plot_loss_curves(train_losses, val_losses)
    plot_accuracy_curves(train_accuracies, val_accuracies)
    
    
    # ===== Evaluation =====
    
    state_dict = torch.load("best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval()

    test_correct = 0
    test_total = 0
    
    # for Confusion Matrix
    all_preds = []
    all_labels = []
    
    
    # for incorrectly predicted images
    wrong_images = []
    wrong_true = []
    wrong_pred = []
    wrong_conf = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
    
            outputs = model(x_batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
    
            # accuracy
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
    
            # confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            # misclassified images
            wrong_mask = preds != y_batch
    
            if wrong_mask.any():
                wrong_images.extend(x_batch[wrong_mask].cpu())
                wrong_true.extend(y_batch[wrong_mask].cpu().numpy())
                wrong_pred.extend(preds[wrong_mask].cpu().numpy())
                wrong_conf.extend(confs[wrong_mask].cpu().numpy())
    
    test_acc = test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    top_images, top_true, top_pred, top_conf = get_top_confident_mistakes(
        wrong_images,
        wrong_true,
        wrong_pred,
        wrong_conf,
        top_n=12
    )
    
    plot_misclassified(top_images, top_true, top_pred, top_conf, CLASS_NAMES)
    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    
    # ===== Confusion Matrix =====
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    plot_confusion_matrix(cm, CLASS_NAMES)
    