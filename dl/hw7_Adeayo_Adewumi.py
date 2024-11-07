# %% [markdown]
# 

# %%
from PIL import Image
import numpy as np
import math
import BitVector
import random
import matplotlib.pyplot as plt
import os
import cv2
import importlib

import torch
import torch.nn as nn
from skimage import io, transform


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
def rgb_to_hsi_pixel(r, g, b):
    # Normalize R, G, B to the range [0, 1]
    R = r / 255.0
    G = g / 255.0
    B = b / 255.0

    # Calculate M, m, and c
    M = max(R, G, B)
    m = min(R, G, B)
    c = M - m

    # Calculate Intensity (I)
    I = (R + G + B) / 3

    # Calculate Saturation (S)
    if I == 0:
        S = 0.0
    else:
        S = 1 - (m / I)

    # Calculate Hue (H)
    if c == 0:
        H = 0.0
    elif M == R:
        H = (60 * ((G - B) / c) + 360) % 360
    elif M == G:
        H = (60 * ((B - R) / c) + 120) % 360
    elif M == B:
        H = (60 * ((R - G) / c) + 240) % 360
    # # Normalize H to the range [0, 1] before returning
    H = H / 360.0

    return H, S, I

def rgb_image_to_hsi(image):
    # Ensure the image is in RGB format (OpenCV loads images in BGR format by default)
    if image.shape[-1] == 3:  # Check if it's a color image with 3 channels
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("The provided image does not have 3 channels (RGB).")

    # Prepare an empty array for HSI with 3 channels (H, S, I)
    hsi_array = np.zeros_like(image_rgb, dtype=float)

    # Iterate over each pixel to convert it to HSI
    for i in range(image_rgb.shape[0]):
        for j in range(image_rgb.shape[1]):
            r, g, b = image_rgb[i, j] / 255.0  # Normalize R, G, B to [0, 1]
            h, s, i_intensity = rgb_to_hsi_pixel(r, g, b)
            # Store the scaled H, S, and I values
            hsi_array[i, j] = [h * 255, s * 255, i_intensity * 255]

    # Extract the Hue channel and scale it to [0, 255]
    hue_channel = hsi_array[:, :, 0]

    # Convert the Hue channel to an 8-bit grayscale image (0-255 range)
    hue_image = Image.fromarray(hue_channel.astype('uint8'), 'L')
    
    return hue_image



# %%


def calculate_lbp(images, R=1, P=8):
    lbp_hists=[]
    for image in images:

        # Extract hue value
        image=rgb_image_to_hsi(image)
        
        # Initialize LBP histogram
        lbp_hist = {t: 0 for t in range(P + 2)}
        
        image = image.resize((64, 64), Image.LANCZOS)
        image=np.array(image)
        width=image.shape[1]
        height=image.shape[0]
        # Constants
        # lbp = [[0 for _ in range(height)] for _ in range(width)]
        rowmax, colmax = height - R, width - R

        # Loop through image pixels
        for i in range(R, rowmax):
            for j in range(R, colmax):
                # print(f"npixel at ({i},{j}):")
                pattern = []

                # Generate pattern for the current pixel using P points
                for p in range(P):
                    # Calculate offsets for circular neighborhood
                    angle = 2 * math.pi * p / P
                    del_k = R * math.cos(angle)
                    del_l = R * math.sin(angle)

                    # Handle very small values close to zero for better stability
                    if abs(del_k) < 0.001: del_k = 0.0
                    if abs(del_l) < 0.001: del_l = 0.0

                    # Calculate neighboring pixel coordinates
                    k = i + del_k
                    l = j + del_l
                    k_base, l_base = int(k), int(l)

                    # Calculate interpolation values
                    delta_k = k - k_base
                    delta_l = l - l_base

                    # Fetch image values and compute the interpolated value at (k, l)
                    image_val_at_p = 0
                    if delta_k < 0.001 and delta_l < 0.001:
                        image_val_at_p = float(image[k_base][l_base])
                    elif delta_k < 0.001:
                        image_val_at_p = (1 - delta_l) * image[k_base][l_base] + delta_l * image[k_base][l_base + 1]
                    elif delta_l < 0.001:
                        image_val_at_p = (1 - delta_k) * image[k_base][l_base] + delta_k * image[k_base + 1][l_base]
                    else:
                        # Bilinear interpolation for fractional (k, l)
                        image_val_at_p = (
                            (1 - delta_k) * (1 - delta_l) * image[k_base][l_base] +
                            delta_k * (1 - delta_l) * image[k_base + 1][l_base] +
                            (1 - delta_k) * delta_l * image[k_base][l_base + 1] +
                            delta_k * delta_l * image[k_base + 1][l_base + 1]
                        )

                    # Append binary pattern based on comparison with center pixel value
                    pattern.append(1 if image_val_at_p >= image[i][j] else 0)

                # print(f"pattern: {pattern}")

                # Convert pattern to BitVector and compute the minimal bit rotation
                bv = BitVector.BitVector(bitlist=pattern)
                intvals_for_circular_shifts = [int(bv << 1) for _ in range(P)]
                minbv = BitVector.BitVector(intVal=min(intvals_for_circular_shifts), size=P)

                # print(f"minbv: {minbv}")

                # Calculate runs of consecutive bits in the minimal rotation
                bvruns = minbv.runs()
                encoding = None

                # Determine encoding based on the number and pattern of runs
                if len(bvruns) > 2:
                    lbp_hist[P + 1] += 1
                    encoding = P + 1
                elif len(bvruns) == 1 and bvruns[0][0] == '1':
                    lbp_hist[P] += 1
                    encoding = P
                elif len(bvruns) == 1 and bvruns[0][0] == '0':
                    lbp_hist[0] += 1
                    encoding = 0
                else:
                    lbp_hist[len(bvruns[1])] += 1
                    encoding = len(bvruns[1])
    lbp_hists.append(lbp_hist)            # print(f"encoding: {encoding}")

    return lbp_hists


# Plot the histogram

def plot_hist(lbp_hists,img_names):
    for i,lbp_hist in enumerate(lbp_hists):
        plt.figure(figsize=(8, 5))
        plt.bar(list(lbp_hist.keys()), lbp_hist.values(), color='g')

        # Labeling the axes and title
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of LBP Patterns for {img_names[i]}')
        plt.savefig(f'lpb_histogram_{img_names[i]}')







# %%
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # encode 1-1
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 1-1
            # encode 2-1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/2

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 2-1
            # encoder 3-1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/4
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 3-1
            # encoder 4-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/8

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 4-1
            # rest of vgg not used
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/16

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 5-1
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True)
        )

    def load_weights(self, path_to_weights):
        vgg_model = torch.load(path_to_weights)
        # Don't care about the extra weights
        self.model.load_state_dict(vgg_model, strict=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # Input is numpy array of shape (H, W, 3)
        # Output is numpy array of shape (N_l, H_l, W_l)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        out = self.model(x)
        out = out.squeeze(0).numpy()
        return out
        

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class CustomResNet(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True):

        super(CustomResNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        # if encoder in ['resnet18', 'resnet34']:
        #     filters = [64, 128, 256, 512]
        # else:
        #     filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        for parameter in resnet.parameters():
            parameter.requires_grad = False

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

    def forward(self, x):
        """
        Coarse and Fine Feature extraction using ResNet
        Coarse Feature Map has smaller spatial sizes.
        Arg:
            x: (np.array) [H,W,C]
        Rerurn:
            xc: (np.array) [C_coarse, H/16, W/16]
            xf: (np.array) [C_fine, H/8, W/8]
        """
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

        x = self.firstrelu(self.firstbn(self.firstconv(x))) #1/2
        x = self.firstmaxpool(x) #1/4

        x = self.layer1(x) #1/4
        xf = self.layer2(x) #1/8
        xc = self.layer3(xf) #1/16

        # convert xc, xf to numpy
        xc = xc.squeeze(0).numpy()
        xf = xf.squeeze(0).numpy()
        return xc, xf




# %%

def get_images(dir, train=True):
    images = []
    labels = []

    # Define the directory path based on the mode (training or testing).
    data_dir = os.path.join(dir, "training" if train else "testing")
    
    # Iterate over sorted image files in the specified directory.
    for img_name in sorted(os.listdir(data_dir)):
        # Skip hidden files like .DS_Store (MacOS metadata files).
        if img_name.startswith('.'):
            continue

        # Identify the label based on the filename.
        label = next((classes.index(cls) for cls in classes if cls in img_name), -1)
        
        # Proceed only if a valid label was found.
        if label != -1:
            # Read the image using cv2.
            img_path = os.path.join(data_dir, img_name)
            img=cv2.imread(img_path)

            # Only add images with three channels (color images).
            if img is not None and img.shape[-1] == 3:
                labels.append(label)
                images.append(img)
    # print(labels)
    return labels, images


def get_features(model, images, labels, mode='train', modelname='vgg', config=None):
    features=[]
    for image in images:
        img=transform.resize(image,(256,256))
        if modelname=='resnet' and config=='coarse':
            feature,_=model(img)
        elif modelname=='resnet' and config=='fine':
            _, feature=model(img)
        else:
            feature=model(img)
        features.append(feature)
    np.savez(f'{modelname}_{mode}_{config}_feature.npz',labels=labels, features=features)



def get_gm(features):
    gm_train = []
    np.random.seed(0)  # Set seed once
    for ft in features:
        ft = ft.reshape(512, -1)
        gm = ft @ ft.T
        gm_flat = gm.flatten()
        
        # Check if enough elements for sampling
        if len(gm_flat) < 1024:
            raise ValueError("Gram matrix is too small for sampling 1024 elements.")
        
        gm_sample = np.random.choice(gm_flat, 1024, replace=False)
        gm_train.append(gm_sample)
        
    return gm_train


def plot_gram_matrix(model, gm_train, training_labels, classes, P, R):
    # Identify one sample index for each class
    sample_indices = []
    for class_name in classes:
        for i, label in enumerate(training_labels):
            if classes[label] == class_name:
                sample_indices.append(i)
                break  # Stop after finding the first match for each class
    # Plot Gram matrices for each identified sample
    for idx, class_name in zip(sample_indices, classes):
        plt.figure()
        gm_2D = gm_train[idx].reshape(32, 32)
        plt.imshow(gm_2D, cmap='viridis')  # Plot Gram matrix of the specific image
        plt.colorbar()
        plt.title(f"{class_name} - 2D Gram Matrix Heatmap ({model}_R={R}_P={P})")
        
        # Save each plot with a unique filename
        plt.savefig(f'{class_name}_2D_Gram_Matrix_Heatmap_{model}_R={R}_P={P}.png')
        plt.show()
  

def plot_confusion_matrix(model, test_labels, pred_labels, R, P):
    # Calculate the confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({model}) R={R}, P={P}")
    plt.savefig(f'confusion_matrix ({model}) R={R}, P={P}.png')


# Define a function to save images of correctly and incorrectly classified samples
def save_classification_images(model, test_images, test_labels, pred_labels, classes, P, R):
    # Initialize dictionaries to track saved images for each class
    saved_correct = {class_name: False for class_name in classes}
    saved_incorrect = {class_name: False for class_name in classes}

    for idx, (image, true_label, pred_label) in enumerate(zip(test_images, test_labels, pred_labels)):
        true_class = classes[true_label]
        pred_class = classes[pred_label]

        # Check if it's correctly or incorrectly classified and save accordingly
        if true_class == pred_class and not saved_correct[true_class]:  # Correctly classified
            plt.figure()
            plt.imshow(image)  # Assuming image is in RGB format
            plt.title(f"Correctly Classified: {true_class}")
            plt.xlabel(f"Predicted: {pred_class} | Ground Truth: {true_class}")
            plt.savefig(f"{model}_correct_{true_class}_P={P}_R={R}.png")
            plt.close()
            saved_correct[true_class] = True  # Mark as saved

        elif true_class != pred_class and not saved_incorrect[true_class]:  # Misclassified
            plt.figure()
            plt.imshow(image)  # Assuming image is in RGB format
            plt.title(f"Misclassified: {true_class} as {pred_class}")
            plt.xlabel(f"Predicted: {pred_class} | Ground Truth: {true_class}")
            plt.savefig(f"{model}_incorrect_{true_class}_P={P}_R={R}.png")
            plt.close()
            saved_incorrect[true_class] = True  # Mark as saved

        # Break the loop once we have both correct and incorrect for each class
        if all(saved_correct.values()) and all(saved_incorrect.values()):
            break





img_dir='HW7-Auxilliary/data'
# *****************************************************************
# *****************************************************************
# *****************************************************************
R, P = 1, 8
model=VGG19()
model.load_weights('HW7-Auxilliary/vgg_normalized.pth')  
classes=['cloudy','rain','shine','sunrise']




training_labels,training_images = get_images(img_dir)
testing_labels, testing_images= get_images(img_dir, train=False)

lbp_train_result=calculate_lbp(training_images, R, P)
np.savez('train_lbp_net.npz', labels=training_labels, features=lbp_train_result)
lbp_test_result=calculate_lbp(testing_images, R, P)
np.savez('test_lbp_net.npz', labels=testing_labels, features=lbp_test_result)

get_features(model, training_images, training_labels)
get_features(model, testing_images, testing_labels, mode='test')

train_data=np.load('vgg_train_None_feature.npz')
test_data=np.load('vgg_test_None_feature.npz')
train_features, train_labels = train_data['features'], train_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']


gm_train = get_gm(train_features)
gm_test = get_gm(test_features)

svm_classifier = SVC(kernel='linear')  
# Fit the model to the training data
svm_classifier.fit(gm_train, train_labels)

# # Make predictions on the test set
pred_labels = svm_classifier.predict(gm_test)

# # Calculate the accuracy score
# train_score = svm_classifier.score(gm_train, train_labels)
test_score = svm_classifier.score(gm_test, test_labels)

## Plot lbp 
# lbp_result=calculate_lbp([cv2.imread('cloudy156.jpg'), cv2.imread('sunrise132.jpg'), cv2.imread('rain67.jpg'), cv2.imread('shine50.jpg')])
# plot_hist(lbp_result, ['cloudy156.jpg', 'sunrise132.jpg','rain67.jpg', 'shine50.jpg'])


print('VGG score',test_score)
plot_confusion_matrix('vgg', test_labels, pred_labels, R, P)
plot_gram_matrix('vgg', gm_train, train_labels, classes, P, R)
save_classification_images('vgg',testing_images, test_labels, pred_labels, classes, P, R)

# *****************************************************************
# *****************************************************************
# *****************************************************************
resnet_config='coarse'
encoder_name='resnet50' # Valid options ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
# CustomResNet will download the model weights from pytorch to the following path:
    # resnet50 ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth  Size - 98MB

resnet = CustomResNet(encoder=encoder_name)
get_features(resnet, training_images, training_labels, modelname='resnet', config=resnet_config)
get_features(resnet, testing_images, testing_labels, mode='test', modelname='resnet', config=resnet_config)

resnet_train_data=np.load('resnet_train_coarse_feature.npz')
resnet_test_data=np.load('resnet_test_coarse_feature.npz')
train_features, train_labels = train_data['features'], train_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']


gm_train = get_gm(train_features)
gm_test = get_gm(test_features)

svm_classifier_resnet = SVC(kernel='linear')  
# Fit the model to the training data
svm_classifier_resnet.fit(gm_train, train_labels)

# # Make predictions on the test set
resnet_pred_labels = svm_classifier_resnet.predict(gm_test)

# # Calculate the accuracy score
# train_score = svm_classifier.score(gm_train, train_labels)
resnet_test_score = svm_classifier_resnet.score(gm_test, test_labels)

## Plot lbp 
# lbp_result=calculate_lbp([cv2.imread('cloudy156.jpg'), cv2.imread('sunrise132.jpg'), cv2.imread('rain67.jpg'), cv2.imread('shine50.jpg')])
# plot_hist(lbp_result, ['cloudy156.jpg', 'sunrise132.jpg','rain67.jpg', 'shine50.jpg'])


print(f'{resnet_config} Resnet score',resnet_test_score)
plot_confusion_matrix(f'{resnet_config}_resnet', test_labels, resnet_pred_labels, R, P)
plot_gram_matrix(f'{resnet_config}_resnet', gm_train, train_labels, classes, P, R)
save_classification_images(f'{resnet_config} Resnet score',testing_images, test_labels, pred_labels, classes, P, R)

# *****************************************************************
# *****************************************************************
# *****************************************************************
resnet_config='fine'
encoder_name='resnet50' # Valid options ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
# CustomResNet will download the model weights from pytorch to the following path:
    # resnet50 ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth  Size - 98MB

resnet = CustomResNet(encoder=encoder_name)
get_features(resnet, training_images, training_labels, modelname='resnet', config=resnet_config)
get_features(resnet, testing_images, testing_labels, mode='test', modelname='resnet', config=resnet_config)

resnet_train_data=np.load('resnet_train_fine_feature.npz')
resnet_test_data=np.load('resnet_test_fine_feature.npz')
train_features, train_labels = train_data['features'], train_data['labels']
test_features, test_labels = test_data['features'], test_data['labels']


gm_train = get_gm(train_features)
gm_test = get_gm(test_features)

svm_classifier_resnet = SVC(kernel='linear')  
# Fit the model to the training data
svm_classifier_resnet.fit(gm_train, train_labels)

# # Make predictions on the test set
resnet_pred_labels = svm_classifier_resnet.predict(gm_test)

# # Calculate the accuracy score
# train_score = svm_classifier.score(gm_train, train_labels)
resnet_test_score = svm_classifier_resnet.score(gm_test, test_labels)

## Plot lbp 
# lbp_result=calculate_lbp([cv2.imread('cloudy156.jpg'), cv2.imread('sunrise132.jpg'), cv2.imread('rain67.jpg'), cv2.imread('shine50.jpg')])
# plot_hist(lbp_result, ['cloudy156.jpg', 'sunrise132.jpg','rain67.jpg', 'shine50.jpg'])


print(f'{resnet_config} Resnet score',resnet_test_score)
plot_confusion_matrix(f'{resnet_config}_resnet', test_labels, resnet_pred_labels, R, P)
plot_gram_matrix(f'{resnet_config}_resnet', gm_train, train_labels, classes, P, R)
save_classification_images(f'{resnet_config} Resnet score',testing_images, test_labels, pred_labels, classes, P, R)


