import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import csv
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

attribute = ['Water', 'Fire', 'Earth', 'Light', 'Dark']
race = ['God', 'Human', 'Demon', 'Beast', 'Dragon', 'Elf', 'Machina', 'Material']
num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
attribute_encoder = preprocessing.LabelEncoder()
attribute_encoder.fit(attribute)
race_encoder = preprocessing.LabelEncoder()
race_encoder.fit(race)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.l1 = nn.Linear(in_features=5, out_features=512, bias=True)
        self.l2 = nn.Linear(in_features=512, out_features=8, bias=True)
    def  forward(self, data):
        data = self.l1(data)
        data = F.dropout(data,p=0.5)
        data = F.leaky_relu(data)
        data = self.l2(data)
        return data

def evaluate(model, X, Y, params = ["acc"]):
    results = []
    predicted = []
    
    inputs = Variable(torch.from_numpy(np.array(X)))
    predicted = model(inputs.double())
    
    predicted = predicted.data.cpu().numpy()
    # CrossEntropyLoss
    predicted = torch.reshape(torch.from_numpy(predicted),(-1,8)).float()
    predicted = torch.argmax(predicted, dim=1)
    
    results = accuracy_score(Y, np.round(predicted))
    return results,

def read_train():
    temp_train = pd.read_csv('train.csv',)
    X = []
    Y = []
    for i in range(len(temp_train)):
        num = []
        for key in num_feats:
            num.append(int(temp_train.iloc[i][key]))
        num.append(attribute_encoder.transform(([temp_train.iloc[i]['Attribute']]))[0])
        X.append(num)
        Y.append(race_encoder.transform([temp_train.iloc[i]['Race']]))
    return X, Y

def read_test():
    temp_test = pd.read_csv('test.csv',)
    X = []
    Y = []
    for i in range(len(temp_test)):
        num = []
        Y.append(temp_test.iloc[i]['id'])
        for key in num_feats:
            num.append(int(temp_test.iloc[i][key]))
        num.append(attribute_encoder.transform(([temp_test.iloc[i]['Attribute']]))[0])
        X.append(num)
    return X, Y

def main():
    net = DNN().double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    
    train_data, train_label = read_train()
    k = int(len(train_data)*0.7)
    train_set = train_data[:k]
    label_set = train_label[:k]
    valid_set = train_data[k:]
    valid_label = train_label[k:]

    running_loss = 0.0
    batch_size = 36
    acc = int(0)
    for epoch in range(0,100001):
        if epoch%10==0:
            print("Epoch ", epoch)
        
        running_loss = 0.0
        for i in range(int(len(train_set)/batch_size)-1):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = torch.from_numpy(np.array(train_set[s:e]))
            labels = torch.from_numpy(np.array(label_set[s:e]))
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.double())
            outputs = torch.reshape(outputs,(-1,8))
            labels = torch.reshape(labels,(-1,))
            
            # CrossEntropyLoss
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        temp = float(list(evaluate(net, valid_set, valid_label))[0])
        if  acc < temp:
            acc = temp

        if epoch%10==0:
            params = ["acc"]
            print (params)
            print ("Training Loss ", running_loss)
            print ("Valid - ", temp)
    print('best acc : ',acc)
    # submit
    net.eval()
    test, test_id = read_test()
    ans = net(torch.from_numpy(np.array(test)).double())
    ans = torch.argmax(ans, dim=1)
    d = {'id': test_id, 'Race': [race_encoder.inverse_transform([ans[i]])[0] for i in range(len(ans))]}
    df = pd.DataFrame(d)
    df[['id', 'Race']].to_csv("submission.csv", index=False)
if __name__ == '__main__':
    main()