import torch
import os
import cv2
import numpy as np
import random


# find maximal sizes (height and width) from all the images
# input_dir - directory containing images
def get_max_sizes(input_dir):
    max_height = 0
    max_width = 0
    files_list = os.listdir(input_dir)
    for file_name in files_list:
        im_path = input_dir + "/" + file_name
        if os.path.isdir(im_path):
            get_max_sizes(im_path)
            continue
        img_cv2 = cv2.imread(im_path)
        im_cur_height = img_cv2.shape[0]
        im_cur_width = img_cv2.shape[1]
        
        print(f' -- File: {file_name}, sizes: {im_cur_height}, {im_cur_width}')
        
        if im_cur_height > max_height:
            max_height = im_cur_height
        if im_cur_width > max_width:
            max_width = im_cur_width
            
    print(f'Directory: {input_dir} - Max Height: {max_height}, Max Width: {max_width}')
    
    
# Select top-N misclassified samples where the model was most confident
# wrong_images - list of misclassified images
# wrong_true   - true labels
# wrong_pred   - predicted (incorrect) labels
# wrong_conf   - confidence scores
# top_n        - number of samples to select
def get_top_confident_mistakes(wrong_images, wrong_true, wrong_pred, wrong_conf, top_n=12):
    indices = sorted(range(len(wrong_conf)), key=lambda i: wrong_conf[i], reverse=True)
    indices = indices[:top_n]

    top_images = [wrong_images[i] for i in indices]
    top_true = [wrong_true[i] for i in indices]
    top_pred = [wrong_pred[i] for i in indices]
    top_conf = [wrong_conf[i] for i in indices]

    return top_images, top_true, top_pred, top_conf


# Set random seed for reproducibility of training results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False