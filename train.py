import abc
import numpy as np
import math
import os
import csv
import CNN
import pickle
import PIL
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sklearn.model_selection as model_selection
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, f1_score



class Model():
    def __init__(self, learning_rate):
        layers = []
        # layers.append(CNN.Convolution_Layer(16, 3, 1, 1,learning_rate))
        # layers.append(CNN.ReLU())
        # layers.append(CNN.MaxPool(2, 2))

        layers.append(CNN.Convolution_Layer(6, 3, 1, 0,learning_rate))
        layers.append(CNN.ReLU())
        layers.append(CNN.MaxPool(2, 2))


        layers.append(CNN.Convolution_Layer(16, 3, 1, 0,learning_rate))
        layers.append(CNN.ReLU())
        layers.append(CNN.MaxPool(2, 2))

        layers.append(CNN.Flatten())
        layers.append(CNN.FullyConnectedLayer(120,learning_rate))
        layers.append(CNN.ReLU())

        layers.append(CNN.FullyConnectedLayer(84,learning_rate))
        layers.append(CNN.ReLU())

        layers.append(CNN.FullyConnectedLayer(10,learning_rate))
        layers.append(CNN.Softmax())

        self.layers = layers


    def train(self,x,y):
        x_copy = np.copy(x)

        print("x_copy",x_copy.shape)

        for layer in self.layers:
            x_copy = layer.forward(x_copy)

        print("x_copy",x_copy.shape)

        # y = np.array(y).reshape(y.shape[0],1)
        # y = np.transpose(y)

        print("genjam hocche")
        
        print("y",y.shape)

        error = (x_copy-y)/y.shape[0]      #error = x-y/y.shape[0] debug here
        
        for layer in reversed(self.layers):
            error = layer.backward(error)

        

        return y,x_copy
        
    def predict(self, x):
        x_copy = x.copy()
        for layer in self.layers:
            x_copy = layer.forward(x_copy)
        return x_copy

    def save(self):
        filename = 'model.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print("Model saved")

    def load(self):
        filename = 'model.pkl'
        path = open(filename,'rb')
        self = pickle.load(path)


#data preprocessing


def load_dataset(image_size, number_of_images): 

    path_a = 'D:\\L4T2\\CSE472\\ass4\\training-a.csv'
    path_b = 'D:\\L4T2\\CSE472\\ass4\\training-b.csv'
    path_c = 'D:\\L4T2\\CSE472\\ass4\\training-c.csv'

    dataframe_forA = pd.read_csv(path_a)
    dataframe_forB = pd.read_csv(path_b)
    dataframe_forC = pd.read_csv(path_c)

    dataframe = pd.concat([dataframe_forA, dataframe_forB, dataframe_forC], ignore_index=True)
    # shuffle dataframe pandas
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    print("Dataframe loaded")


    images = 'D:\\L4T2\\CSE472\\ass4\\' + dataframe['database name'] + '\\' + dataframe['filename'] # \\ na diye / deya chilo.still didn't work
    images = images[:number_of_images]
    y = dataframe['digit'][:number_of_images]

    count = len(images)
    index = 0

    images_output = []

    for i in images:
        # load images using cv2
        i = cv2.imread(i)
        i = cv2.resize(i, image_size)
        i = (255.0 - np.array(i))/255.0

        images_output.append(i)


    input = np.array(images_output)
    #input = np.transpose(input, (0, 3, 1, 2)) #
    return input,y

def train_split(x,y,split):
    x_train = x[:int(x.shape[0]*split),:,:,:]
    #y_train = y[:int(y.shape[0]*split)]
    y_train = y[:int(y.shape[0]*split)][:, np.newaxis]
    x_valid = x[int(x.shape[0]*split):,:,:,:]
    y_valid = y[int(y.shape[0]*split):][:, np.newaxis]
    #y_valid = y[int(y.shape[0]*split):]
    return x_train, y_train, x_valid, y_valid

#test_load_dataset here
def testLoadDataset(path, image_size):
    


    images_paths = [k for k in os.listdir(path) if os.path.isfile(os.path.join(path, k))]

    images_output = []

    for i in images_paths:
        # load images using cv2
        i = cv2.imread(i)
        i = cv2.resize(i, image_size)
        i = (255.0 - np.array(i))/255.0

        images_output.append(i)


    input = np.array(images_output)
    # input = np.transpose(input, (0, 3, 1, 2)) #
    return images_paths,input






#main functions are here


def Loss(y_true, y_pred):
    loss = 0
    for i in range(y_pred.shape[0]):
        loss += y_pred[i] - y_true[i]
    return loss/y_pred.shape[0]


def Accuracy(y_true, y_pred):
    acc = 0
    for i in range(y_pred.shape[0]):
        if np.argmax(y_pred[i]) == y_true[i]:
            acc += 1
    return acc/y_pred.shape[0]

def Macro_f1(y_true, y_pred):
    f1 = 0
    for i in range(y_pred.shape[0]):
        if np.argmax(y_pred[i]) == y_true[i]:
            f1 += 1
    return f1/y_pred.shape[0]

def Confusion_matrix(y_true, y_pred):
    cm = np.zeros((10,10))
    for i in range(y_pred.shape[0]):
        cm[y_true[i]][np.argmax(y_pred[i])] += 1
    return cm

# def train_final(model, x, y, batch_size, epochs,learning_rate):

#     x_train, y_train, x_valid, y_valid = train_split(x,y,0.8)
#     #x_train , y_train , x_valid , y_valid = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
#     file = open('results.csv', 'w')
#     writer = csv.writer(file)
#     writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_acc','macro_f1','confusion_matrix'])
#     all_train_loss = []
#     all_valid_loss = []
#     all_valid_acc = []
#     all_macro_f1 = []
#     all_confusion_matrix = []

#     for epoch in range(epochs):
#         train_loss = 0
#         valid_loss = 0
#         train_acc = 0
#         valid_acc = 0
#         macro_f1 = 0
#         confusion_matrix = 0
#         csvFileRows = []
#         csvFileRows.append(epoch)
#         batch_count = 0
        
#         for i in range(0, x_train.shape[0], batch_size):
#             x_batch = x_train[i:i+batch_size]
#             y_batch = y_train[i:i+batch_size]
#             print("x_batch shape: ",x_batch.shape)
#             print("y_batch shape: ",y_batch.shape)

            
#             y_true = np.zeros((10, batch_size))

#             print("y_true shape before hot: ",y_true.shape)


#             for k in range(y_true.shape[0]):
#                 y_true[y_train[k],k] = 1

#             print("y_true shape: ",y_true.shape)

#             y_output, y_pred = model.train(x_batch, y_true)

#             #y_output, y_pred = model.train(x_batch, y_batch) 

#             y_true = np.zeros((10,y_output.shape[0]))

#             # for j in range(y_output.shape[0]):
#             #     y_true[y_output[j],j] = 1
            
#             # y_pred = np.transpose(y_pred)
#             print(y_pred.shape)
#             print(y_true.shape)
#             batch_count += 1

#             # y_true_ = y_true(axis=1)
#             # y_pred_ = y_pred.argmax(axis=1)


#             # train_loss += metrics.log_loss(y_output, y_pred)
#             # train_acc += Accuracy(np.argmax(y_prediction,axis= 0),np.argmax(y_true,axis=0))
#             # macro_f1 += Macro_f1(np.argmax(y_prediction,axis= 0),np.argmax(y_true,axis=0))
#         print("Training Estimation:")
#         print("batch_count: ", batch_count)
#         # print("Train Loss: ", train_loss/batch_size)
#         #loss_append = (train_loss/batch_size)[batch_count]
#         # csvFileRows.append(train_loss/batch_size)
#         # print("Train Accuracy: ", train_acc/batch_size)
#         # print("Train Macro F1: ", macro_f1/batch_size)

#         #Validation

#         print("Validation Estimation:")
#         y_true = np.zeros((len(y_valid),10))

#         # print('y_valid shape: ', y_valid.shape)

#         y_pred = model.predict(x_valid)
#         print('model.predict(x_valid) run hoyeche')

#         for i in range(y_true.shape[1]):
#             y_true[y_valid[i][i, 0]] = 1
#         # y_pred = np.transpose(y_pred)
#         print(y_pred.shape)
#         print(y_true.shape)

#         # for j in range(len(y_valid)):
#         #     y_true[y_valid[j],j] = 1
        
#         valid_loss = metrics.log_loss(y_valid, y_pred)
#         valid_acc = metrics.accuracy_score(np.argmax(y_true,axis= 0),np.argmax(y_pred,axis=0))
#         macro_f1 = metrics.f1_score(np.argmax(y_true,axis= 0),np.argmax(y_pred,axis=0),average='macro')
#         confusion_matrix = metrics.confusion_matrix(np.argmax(y_true,axis= 0),np.argmax(y_pred,axis=0))

#         csvFileRows.append(valid_loss)
#         csvFileRows.append(valid_acc)
#         csvFileRows.append(macro_f1)
#         csvFileRows.append(confusion_matrix)

#         print("Valid Loss: ", valid_loss)
#         print("Valid Accuracy: ", valid_acc)
#         print("Valid Macro F1: ", macro_f1)
#         print("Confusion Matrix: ", confusion_matrix)

#         all_train_loss.append(train_loss/batch_size)
#         all_valid_loss.append(valid_loss)
#         all_valid_acc.append(valid_acc)
#         all_macro_f1.append(macro_f1)
#         # all_confusion_matrix.append(confusion_matrix)

#         writer.writerow(csvFileRows)

#     #plots here
#     print("Training Finished")
#     plt.plot(all_train_loss, label='train_loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Train Loss')
#     plt.savefig('train_loss.png')

#     plt.plot(all_valid_loss, label='valid_loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Valid Loss')
#     plt.savefig('valid_loss.png')

#     plt.plot(all_valid_acc, label='valid_acc')
#     plt.xlabel('Epochs')
#     plt.ylabel('Valid Accuracy')
#     plt.savefig('valid_acc.png')

#     plt.plot(all_macro_f1, label='macro_f1')
#     plt.xlabel('Epochs')
#     plt.ylabel('Macro F1')
#     plt.savefig('macro_f1.png')

    # plt.plot(all_confusion_matrix, label='confusion_matrix')
    # plt.xlabel('Epochs')
    # plt.ylabel('Confusion Matrix')
    # plt.show()
    # plt.savefig('confusion_matrix.png')



# train model having X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size, y is one hot encoded


def train_model(model,x,y,batch_size,epochs,learning_rate):
    X_train , y_train , X_val , y_val = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
# initialize loss and accuracy lists
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        # shuffle training data
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # get number of batches
        num_batches = len(X_train) // batch_size
        loss = 0
        acc = 0
        for i in tqdm(range(num_batches)):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]
            # print(X_batch.shape, y_batch.shape)
            
            # forward pass
            out = X_batch
            for layer in model:
                out = layer.forward(out)
            # print(f'out.shape: {out.shape}')
            # print(f'out: {out}')
            
            loss += log_loss(y_batch, out)

            # calculate accuracy
            acc += accuracy_score(np.argmax(y_batch, axis=1), np.argmax(out, axis=1))

            # backward pass
            dL_dout = np.copy(out)
            dL_dout -= y_batch
            dL_dout /= batch_size
            for layer in reversed(model):
                dL_dout = layer.backward(dL_dout, learning_rate)
            
        train_loss.append(loss/num_batches)
        train_acc.append(acc/num_batches)
        # validation
        
        val_out = X_val
        for layer in model:
            val_out = layer.forward(val_out)
        val_loss.append(log_loss(y_val, val_out))
        val_acc.append(accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_out, axis=1)))
        val_f1.append(f1_score(np.argmax(y_val, axis=1), np.argmax(val_out, axis=1), average='macro'))

        print(f'Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.4f}')
        print(f'Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}')
        print(f'Val F1: {val_f1[-1]:.4f}')


    

np.random.seed(0)
batch_size = 32
epochs = 50
learning_rate = 0.001
number_of_images = 1000



x,y = load_dataset((20,20), number_of_images)

model = Model(learning_rate)

train_model(model, x, y, batch_size, epochs, learning_rate)
model.save()


# try:
#     train(model, x, y, batch_size, epochs, learning_rate)
# except:
#     print('Error')
#     model.save()
#     exit(0)

    
        
