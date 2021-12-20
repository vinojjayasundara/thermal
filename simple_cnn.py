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

parser = argparse.ArgumentParser(description='Simple CNN Classification')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--nepoch', default=10, type=int, help='Number of epochs')

args = parser.parse_args()

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
    
class TrainTest():
    
    def __init__(self,args, model, trainset, testset):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
          
        self.model.to('cuda')          

        self.args = args
        self.chkpath = './results/cnn.pt'

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
            plot_losses(self.state['details'], 'cnn')
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
        
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = self.model(inputs)
                loss = F.nll_loss(outputs, labels)
                loss = loss.item() 
                running_loss = running_loss + loss
                
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                count += outputs.shape[0]        
        return correct/count*100., running_loss/count, correct

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
                outputs = self.model(inputs)
                with torch.no_grad():
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                loss = F.nll_loss(outputs, labels)
                loss.backward()
                if i%10==0 and i>1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                loss = loss.item() 
                running_loss = running_loss + loss
                count = count + outputs.shape[0]
            TRAIN_LOSS =  running_loss/count  
            TRAIN_ACC = correct/count*100.
           
            TEST_ACC, TEST_LOSS, TEST_COUNT = self.test()
            
            self.details.append((TRAIN_LOSS,TRAIN_ACC,TEST_LOSS,TEST_ACC))

            if TEST_ACC > self.best_acc:                
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
            print('[{}] [{:.1f}] [Loss {:.3f}] [Correct : {}] [Trn. Acc {:.1f}] '.format(epoch, elapsed_time,
                    TRAIN_LOSS, correct,TRAIN_ACC),end=" ")
            print('[Test Cor {}] [Loss {:.3f}] [Acc {:.1f}]'.format(TEST_COUNT, TEST_LOSS, TEST_ACC))   
            
        plot_losses(self.state['details'],'cnn')


def main(args):    
    
    trainset = Therm_Dataset('./npy_dataset',is_train=True)
    testset = Therm_Dataset('./npy_dataset',is_train=False)
    
    model = CNN(num_classes = 2)
    
    summary(model, (300, 224, 224))
            
    EXEC = TrainTest(args,model=model, trainset=trainset, testset = testset)
    EXEC.train()
    
main(args)