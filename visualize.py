"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import cv2

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


raw_img = io.imread('images/1.jpg')
print(raw_img.shape)
gray = rgb2gray(raw_img)
print(gray.shape)

gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

print(gray.shape)

img = gray[:, :, np.newaxis]

print(img.shape)

img = np.concatenate((img, img, img), axis=2)
print(img.shape)
img = Image.fromarray(img)
print(img.size)
print(img.size)
inputs = transform_test(img)
print(inputs.shape)
inputs = torch.unsqueeze(inputs[0], 0)
# print(inputs[0][0])
# print(inputs[0][1])
print(inputs.shape)

image = cv2.imread("images/1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (44, 44))
image_tensor = torch.from_numpy(image)
print(image_tensor.shape)
input = torch.stack([image_tensor, image_tensor, image_tensor])
print(input.shape)
input = torch.unsqueeze(input,0)
print(input.shape)
input = input.float()
input = input.cuda()

print(inputs.dtype)
print(input.dtype)

class_names = ['Angry', 'Disgust', 'Fear',
               'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

# ncrops, c, h, w = np.shape(inputs)

# inputs = inputs.view(-1, c, h, w)
inputs = inputs.cuda()
inputs = Variable(inputs, volatile=True)
outputs = net(input)

# outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

score = F.softmax(outputs)
print(score)
_, predicted = torch.max(outputs.data[0], 0)

# plt.rcParams['figure.figsize'] = (13.5,5.5)
# axes=plt.subplot(1, 3, 1)
# plt.imshow(raw_img)
# plt.xlabel('Input Image', fontsize=16)
# axes.set_xticks([])
# axes.set_yticks([])
# plt.tight_layout()


# plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

# plt.subplot(1, 3, 2)
# ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
# width = 0.4       # the width of the bars: can also be len(x) sequence
# color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
# for i in range(len(class_names)):
#     plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
# plt.title("Classification results ",fontsize=20)
# plt.xlabel(" Expression Category ",fontsize=16)
# plt.ylabel(" Classification Score ",fontsize=16)
# plt.xticks(ind, class_names, rotation=45, fontsize=14)

# axes=plt.subplot(1, 3, 3)
# emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
# plt.imshow(emojis_img)
# plt.xlabel('Emoji Expression', fontsize=16)
# axes.set_xticks([])
# axes.set_yticks([])
# plt.tight_layout()
# # show emojis

# #plt.show()
# plt.savefig(os.path.join('images/results/l.png'))
# plt.close()

print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))
