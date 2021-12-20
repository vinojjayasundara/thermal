import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import os
import time
import argparse
from utils import Therm_Dataset, plot_losses
from torchviz import make_dot
from torch.autograd import Variable
from torchsummary import summary
import torchvision
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics

parser = argparse.ArgumentParser(description='Simple Autoencoder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--nepoch', default=25, type=int, help='Number of epochs')

args = parser.parse_args()

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
    
class TrainTest():
    
    def __init__(self,args, model, trainset, testset):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
          
        self.model.to('cuda')          

        self.args = args
        self.chkpath = './results/autoencoder.pt'

        if not os.path.exists('results/'):
            os.mkdir('results/')
            
        print(self.chkpath)
        if os.path.exists(self.chkpath) == True:
            print('load from results', end=' ')
            self.state = torch.load(self.chkpath)
            self.model.load_state_dict(self.state['model'])
            best_acc = self.state['acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch == self.args.nepoch:
                print('existing as epoch is max.')
            plot_losses(self.state['details'],'autoencoder')
            self.details = self.state['details']    
            self.best_acc = best_acc
            self.start_epoch = start_epoch + 1
            self.model.to('cuda')          
        else:
            self.best_acc = -1.
            self.details = []   
            self.start_epoch = 0
            
        cudnn.benchmark = True
        
    def test(self):
        self.model.eval()
        correct = 0
        count = 0
        running_loss = 0.0
        latent_vals = np.empty([0,256])
        latent_labels = np.empty([0])
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                _, outputs = self.model(inputs)
                numpy_outputs = outputs.cpu().detach().numpy()
                numpy_labels = labels.cpu().detach().numpy()
                latent_vals = np.concatenate([latent_vals,numpy_outputs.reshape((1,-1))])
                latent_labels = np.concatenate([latent_labels,numpy_labels])
                count += outputs.shape[0]       
                
            indices = [15,16,17,18,19,20,21,22,23,9,10,11,12,13,14]
            x_train = latent_vals[indices]
            y_train = latent_labels[indices]

            indices = [0,1,2,3,4,5,6,7,8,24,25,26,27,28,29]
            x_test = latent_vals[indices]
            y_test = latent_labels[indices]

            # Linear Kernel Classifier
            classifier = SVC(kernel='linear')
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)

            print('Accuracy for {} classifier:'.format('linear'), accuracy)
            print('Precision for {} classifier:'.format('linear'), precision)
            print('Recall for {} classifier:'.format('linear'), recall)
        return 0., running_loss/count, 0.

    def train(self):     
        for epoch in range(self.start_epoch,self.args.nepoch):  
            self.model.train()
            start_time = time.time()        
            running_loss = 0.0
            correct = 0.
            count = 0.
            self.optimizer.zero_grad()            
            for i, data in enumerate(self.trainloader, 0):        
                inputs, labels = data 
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                if i%1==0 and i>1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                loss = loss.item() 
                running_loss = running_loss + loss
                count = count + outputs.shape[0]
            TRAIN_LOSS =  running_loss/count  
            TRAIN_ACC = 0
            TEST_ACC, TEST_LOSS, TEST_COUNT = 0, 0, 0
            
            self.details.append((TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC))

            if TEST_ACC >= self.best_acc:                
                self.state = {
                    'model': self.model.state_dict(),
                    'acc': TEST_ACC,
                    'epoch': epoch,
                    'details':self.details,            
                    'args_list':self.args,
                }        
                torch.save(self.state, self.chkpath)
                self.best_acc = TEST_ACC
            else:
                self.state['epoch'] = epoch
                torch.save(self.state, self.chkpath)
            elapsed_time = time.time() - start_time
            print('[{}] [{:.1f}] [Loss {:.3f}]'.format(epoch, elapsed_time, TRAIN_LOSS))            
        plot_losses(self.state['details'],'autoencoder')


def main(args):    
    
    trainset = Therm_Dataset('./npy_dataset',is_train=True, auto=True)
    testset = trainset
    model = AutoEncoder()
    
    summary(model, (300, 224, 224))
            
    EXEC = TrainTest(args,model=model, trainset=trainset, testset = testset)
    EXEC.train()
    EXEC.test()
    
    
main(args)