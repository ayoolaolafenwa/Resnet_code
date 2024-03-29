import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
#import accuracy from torchfusion utils
from torchfusion_utils.metrics import Accuracy

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size =3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output) 
        output = self.relu(output)   

        return output



class Custom_class(nn.Module):
    def __init__(self, num_classes = 10):
        super(Custom_class, self).__init__()

        self.net = nn.Sequential(
            
          Unit(in_channels = 3, out_channels = 32),
          Unit(in_channels = 32, out_channels = 32),
          Unit(in_channels = 32, out_channels = 32),
          Unit(in_channels = 32, out_channels = 32),

          nn.MaxPool2d(kernel_size = 2,stride=2),

          Unit(in_channels = 32, out_channels = 64),
          Unit(in_channels = 64, out_channels = 64),
          Unit(in_channels = 64, out_channels = 64),
          Unit(in_channels = 64, out_channels = 64),

          nn.MaxPool2d(kernel_size = 2, stride=2),

          Unit(in_channels = 64, out_channels = 128),
          Unit(in_channels = 128, out_channels = 128),
          Unit(in_channels = 128, out_channels = 128),
          Unit(in_channels = 128, out_channels = 128),

          nn.MaxPool2d(kernel_size = 2, stride=2),

          Unit(in_channels = 128, out_channels = 256),
          Unit(in_channels = 256, out_channels = 256),
          Unit(in_channels = 256, out_channels = 256),

          nn.AvgPool2d(kernel_size = 4)
        )

        self.fc =  nn.Linear(in_features = 256, out_features = num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 256)
        output = self.fc(output)
        return output


train_transformations = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(32, padding = 4),
  transforms.ToTensor(),
  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]) 



train_set = CIFAR10(root = "./data", train = True, transform = train_transformations, download = True)
batch_size = 32


train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)


test_transformations = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_set = CIFAR10("./data",train = False, transform = test_transformations, download = True)

test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 4)

#check gpu support is available
cuda_avail =torch.cuda.is_available()

#create optimizer, model and lossfunction
model = Custom_class(num_classes = 10)

if cuda_avail:
  model.cuda()

optimizer = Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)    
loss_fn = nn.CrossEntropyLoss()

def lr_rate(epoch):
  lr = 0.001
  if(epoch > 5):
      lr = lr/5
  elif(epoch > 8):
      lr = lr/8    
  for param_group in optimizer.param_groups:
      param_group["lr"] = lr

def save_models(epoch):
  torch.save(model.state_dict, "cifar10model_{}.model".format(epoch)) 
  print("checkpoint saved")   

def test():
  model.eval()
  #create an instance of the accuracy metric
  test_metric = Accuracy()
  for a, (images, labels) in enumerate(test_loader):

    if cuda_avail:
      images = images.cuda()
      labels = labels.cuda()

      #Predict classes using images from the test set
      outputs = model(images)
      #update the accuracy
      test_metric.update(outputs,labels)

  #return the test accuracy
  return test_metric.getValue()

def train(num_epochs):

  best_acc = 0.0

  for epoch in range(num_epochs):

      model.train()
      #create an instance of the accuracy metric
      train_metric = Accuracy()
      
      train_loss = 0.0
      for a , (images, labels)in enumerate(train_loader):
          
        #move the labels and images to gpu 
        if cuda_avail:

            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        #Compute the loss based on the predictions and actual labels
        loss = loss_fn(outputs, labels)
        #Backpropagate the loss
        loss.backward()
        #Adjust parameters according to the computed gradients
        optimizer.step()
        train_loss += loss.item() * images.size(0)

        #update the training accuracy
        train_metric.update(outputs,labels)
      
      
      #Call the learning rate adjustment function
      lr_rate(epoch) 

      #Compute the average acc and loss over all 50000 training images
      train_loss = train_loss/50000

      #Evaluate on the test set
      test_acc = test()

      #get the value of the training accuracy
      train_acc = train_metric.getValue()

      # Save the model if the test acc is greater than our current best
      if test_acc > best_acc:

          save_models(epoch)
          best_acc = test_acc
      print("Epoch{}, Train_acc: {}, Train_loss: {}, Test_acc: {}".format(epoch, train_acc, train_loss, test_acc))    

train(200)