import sys
import socket
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt



# ML MODEL
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
    def __init__(self, client_id, model, learning_rate, batch_size):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(client_id)
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


# SERVER
class Server:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = "Server"
        self.PORT = int(sys.argv[1]) #(PORT 6000)
        self.sub_client = sys.argv[2] #(sub_client)

        self.clients = {}  #{"client1": <data_size>, "client2": <data_size>...}
        self.server_model = MCLR()

        self.num_glob_iters = 100
        self.learning_rate = 0.01


    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                print("Server host names: ", self.ID, " Port: ", self.PORT)
                s.bind((self.HOST, self.PORT))        # Bind to the port
                s.listen(5)
                while(True):
                    c, addr = s.accept()     #1. Establish connection with client.
                    print("[Server] Got connection from", addr)

                    #2. Receive data from socket..
                    data = c.recv(1024)
                    
                    if not data:
                        return
                    
                    dic = data.decode('utf-8')
                    data_json = json.loads(dic) #{"id": "client1", "data_size": 99}

                    if self.clients.get(data_json["id"], None) == None:
                        self.clients[data_json["id"]] = data_json["data_size"]


                    print("FIRST HANDSHAKE COMPLETE.. waiting 30 seconds")
                    time.sleep(30)

        except Exception as e:
            print(['[Server] ', e])
            print("[Server] Can't connect to the Socket: " + str(e))


    #Broadcast global model to all clients
    def send_parameters(self, server_model):
        for c in self.clients:
            c.set_parameters(server_model)

    #Aggregate local models from clients into global model
    def aggregate_parameters(self, server_model, total_train_samples):
        # Clear global model before aggregation
        for param in server_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for c in self.clients:
            for server_param, user_param in zip(server_model.parameters(), c.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * c.train_samples / total_train_samples
        return server_model

    #evaluate global model across all clients
    def evaluate(self, client):
        total_accurancy = 0
        for c in self.clients:
            total_accurancy += client.test()
        return total_accurancy/len(self.clients)



server = Server()
server.run()