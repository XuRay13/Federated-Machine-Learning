import sys
import time
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

import pickle

class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor. 
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size, data):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = data
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.trainloader = DataLoader(self.train_data, self.train_samples)
        self.testloader = DataLoader(self.test_data, self.test_samples)

        self.loss = nn.NLLLoss()

        self.model = copy.deepcopy(model)

        self.id = client_id

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                print(len(X))
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print(str(self.id) + ", Accuracy of client ",self.id, " is: ", test_acc)
        return test_acc




class Client:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = sys.argv[1] # (client1)
        self.ID_num = int(sys.argv[1][-1]) #(1)
        self.PORT = int(sys.argv[2]) #(PORT 6001)

    def run(self):
        X_train, y_train, X_test, y_test, train_samples, test_samples = self.get_data()
        

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.connect((self.HOST, 6000)) #"A", 6000
                print("established connection to server")

                # Send handshake
                msg = {"id": self.ID_num, "type": "handshake", "data_size": len(X_train)}

                msg_data_json = json.dumps(msg)
                s.send(msg_data_json.encode('utf-8'))
                print("[Client{}] sent data...".format(self.ID))

            
            while True:
                print("I am client " + str(self.ID_num))
                print("Receiving new global model")
                # Receive global model
                model_data = self.listen_for_broadcast()

                print("received model pickle string: " + str(dir(model_data)))
                # TODO: Test local model
                
                
                # TODO: Train local model
                # Send local model
                msg = {"id": self.ID_num, "type": "model", "model": None}

                msg_data_json = json.dumps(msg)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.connect((self.HOST, 6000)) #"A", 6000
                    s.send(msg_data_json.encode('utf-8'))
                    s.close()
        
        except Exception as e:
            print("[Client] Can't connect to the Server:  ", str(e))
            exit()

    def listen_for_broadcast(self):
        # Simply listen for a message from the server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.HOST, self.PORT))
                s.listen(1) # listen for server message
                c, addr = s.accept()
                data = c.recv(65536)
                if not data:
                    return
                # data_json = json.loads(data.decode('utf-8'))
                server_model = pickle.loads(data) 
                return server_model
            
        except Exception as e:
            print("Failed to open socket as server: " + str(e))

    def test_model(self):
        # TODO
        pass
        # self.model.eval()
        # test_acc = 0
        # for x, y in self.testloader:
        #     output = self.model(x)
        #     test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
        #     print(str(self.id) + ", Accuracy of client ",self.id, " is: ", test_acc)
        # return test_acc

    def train_model(self):
        # TODO
        pass


    def get_data(self, id=""):
        train_path = os.path.join("FLdata", "train", "mnist_train_" + self.ID + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_" + self.ID + ".json")
        train_data = {}
        test_data = {}

        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data'])

        X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
        X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_samples, test_samples = len(y_train), len(y_test)
        return X_train, y_train, X_test, y_test, train_samples, test_samples
    
    def test(self):
        X_train, y_train, X_test, y_test, _, _= self.get_data()
        print(len(X_train))
        print(len(y_train))
        image = X_train[0].numpy().reshape(28,28)
        print(image.shape)
        plt.imshow(image, cmap='gray')
        print("label:",y_train[0])

    


client = Client()
client.run()
x = client.test()


