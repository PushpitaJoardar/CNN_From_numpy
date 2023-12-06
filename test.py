import sys
import numpy as np
import csv
import pickle
import os
import cv2
from train import Model
from train import Convolution_Layer
from train import ReLU
from train import MaxPool
from train import Flatten
from train import FullyConnectedLayer
from train import Softmax
from sklearn.metrics import accuracy_score, log_loss, f1_score

learning_rate = 0.001
number_of_images = 2727


def load_images(folder_path, count):
    images = []
    i = 0
    for filename in os.listdir(folder_path):
        # load the image using OpenCV grey scale
        img = cv2.imread(os.path.join(folder_path, filename),0)
        # compress image pixels
        img = cv2.resize(img, (28, 28))
        # convert the image to a NumPy array of shape (channel, height, width)
        img = (255 - np.array([img])) / 255

        images.append(img)
        i +=1
        if i == count:
            break
    return images


x = load_images('D:/L4T2/CSE472/1705052/test-a2', number_of_images)
# y = np.loadtxt('D:/L4T2/CSE472/ass4/training-d.csv', delimiter=',', skiprows=1, usecols=2, max_rows= number_of_images, dtype=int)
# image_names = np.loadtxt('D:/L4T2/CSE472/ass4/training-d.csv', delimiter=',', skiprows=1, usecols=6, max_rows= number_of_images, dtype=str)



print("data loaded")
# print(len(y))

def load_model():
    filename = 'model.pkl'
    path = open(filename,'rb')
    return pickle.load(path)


x = np.array(x)
x = x.transpose(0, 2, 3, 1)
# y = np.array(y)


print("LOadDATASET")
# model = train.Model(0.001) #learning rate
# model.load('model.pkl')
model = load_model()

out = model.trainForward(x)

# print('debug')
print(out.shape)
# print(y.shape)

# # y to one hot
# y = np.eye(10)[y]

# loss = log_loss(y, out)

# # # calculate accuracy
# acc = accuracy_score(np.argmax(y, axis=1), np.argmax(out, axis=1))

# f1 = f1_score(np.argmax(y, axis=1), np.argmax(out, axis=1), average='macro')

# print('loss: ', loss)
# print('accuracy: ', acc)
# print('f1_score: ', f1)

prediction = np.argmax(out, axis=1)
print(prediction.shape)
# print(prediction)

print("prediction done")

with open('1705052_prediction.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['FileName', 'Digit'])
    for i in range(len(prediction)):
        writer.writerow([i, prediction[i]])
