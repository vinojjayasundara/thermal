import torch
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import gzip
from PIL import Image
import numpy as np
import os
import glob


class Therm_Dataset(dataset.Dataset):

    def __init__(self, dataset_path, is_train = True, auto = False, transform=None):
        """
        Args:
            dataset_path (string): Path to the training/testing images.
            is_train (bool): Training or testing dataset.
            auto (bool): Load for the AutoEncoder when True, for the CNN otherwise.
            transform (callable): Any data augmentations.
        """
        im_size = 224
        label_set = ["NM","M"]
        samples = glob.glob(os.path.join(dataset_path,'*','*','*.npy'))
        
        # Load all data
        entire_set = np.zeros([len(samples),100,im_size,im_size,3])
        entire_labels = np.zeros(len(samples))
        
        for idx, samp in enumerate(samples):
            current_sample = np.load(samp)
            label = label_set.index(samp.split('/')[2])
            entire_set[idx] = current_sample
            entire_labels[idx] = label
        entire_set = np.transpose(entire_set,(0,2,3,1,4))
        entire_set = np.reshape(entire_set,(len(samples),im_size,im_size,-1))
        entire_set = np.transpose(entire_set,(0,3,1,2))
        self.images = torch.tensor(entire_set,dtype=torch.float32)
        self.labels = torch.tensor(entire_labels,dtype=torch.long)

        if not auto:
            if is_train:
    #             indices = [0,1,2,3,4,5,6,7,8,24,25,26,27,28,29]
                indices = [15,16,17,18,19,20,21,22,23,9,10,11,12,13,14]
                self.images = self.images[indices]
                self.labels = self.labels[indices]
            else:
    #             indices = [15,16,17,18,19,20,21,22,23,9,10,11,12,13,14]
                indices = [0,1,2,3,4,5,6,7,8,24,25,26,27,28,29]
                self.images = self.images[indices]
                self.labels = self.labels[indices]
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
    
def plot_losses(details, prefix='autoencoder'):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for det in details:
        train_loss.append(det[0])
        train_acc.append(det[1])
        test_loss.append(det[2])
        test_acc.append(det[3])
        
    plt.figure()
    plt.plot(list(range(1,len(train_loss)+1)),train_loss, 'r', label='train')
    plt.plot(list(range(1,len(test_loss)+1)),test_loss, 'b', label='test')
    plt.plot(list(range(1,len(train_loss)+1)),train_loss, 'rx')
    plt.plot(list(range(1,len(test_loss)+1)),test_loss, 'bx')
    plt.legend()
    plt.title('Loss curves for train and test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/{}_loss_curves.png'.format(prefix))
    plt.close()
    
    plt.figure()
    plt.plot(list(range(1,len(train_acc)+1)),train_acc, 'r' ,label='train')
    plt.plot(list(range(1,len(test_acc)+1)),test_acc , 'b' ,label='test')
    plt.plot(list(range(1,len(train_acc)+1)),train_acc,'rx')
    plt.plot(list(range(1,len(test_acc)+1)),test_acc ,'bx')
    plt.title('Accuracy curves for train and test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/{}_acc_curves.png'.format(prefix))
    plt.close()
    
    
class CNN(nn.Module):
    def __init__(self, num_classes):        
        super(CNN,self).__init__()    
        self.num_classes = num_classes         
        self.conv1 = nn.Conv2d(300, 8, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(32, 64, 5, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)      
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self):        
        super(AutoEncoder,self).__init__()    
        self.encoder = nn.Sequential(
            nn.Conv2d(300, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 4, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 4, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 300, 6, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec, enc