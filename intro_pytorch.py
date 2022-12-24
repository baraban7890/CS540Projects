import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import normalize
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)

    if(training == True):
        return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size = 64)

def build_model():
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10)
)
    return model




def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print("Train Epoch: "+str(epoch)+ "   Accuracy: " + str(correct)+ "/"+ str(total) + "("+'{:.2f}'.format(correct*100/total)+"%)   Loss:"+'{:.3f}'.format(running_loss/len(train_loader.dataset)))
    return

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
    if(show_loss == True):
        print("Average loss: "+'{:.4f}'.format(running_loss/len(test_loader.dataset)))
    print("Accuracy: "+'{:.2f}'.format(correct*100/total)+"%")
    return
    


def predict_label(model, test_images, index):
    logits = model(test_images[index])
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    with torch.no_grad():
        for data in test_images:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1
    prob = F.softmax(logits, dim=1)
    for x in range(3):
        print(str(class_names[torch.argmax(prob)]) + ": " + '{:.2f}'.format(torch.max(prob)*100)+"%")
        prob[0,torch.argmax(prob)] = 0


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    model = build_model()
    print(model)
    train_model(model,train_loader,criterion,5)
    evaluate_model(model,test_loader,criterion)
    