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

lr = 0.01
# Check 5, 10, 20
batch_size = 20
E = 2

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
        if batch_size:
            self.trainloader = DataLoader(self.train_data, batch_size)
        else:
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
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data
    
    def train_loss(self):
        self.model.train()
        for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
        print("Training loss: {}".format(str(loss.data.item())))        
        return loss.data.item()
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print("Testing accuracy: {0:.2f}%".format(test_acc * 100))
        return test_acc




class Client:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = sys.argv[1] # (client1)
        self.ID_num = int(sys.argv[1][-1]) #(1)
        self.PORT = int(sys.argv[2]) #(PORT 6001)

        self.mini_batch_GD = int(sys.argv[3])

    def run(self):
        data = self.get_data()
        

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.connect((self.HOST, 6000)) #"A", 6000

                # Send handshake
                msg = {"id": self.ID_num, "type": "handshake", "data_size": len(data[0])}

                msg_data_json = json.dumps(msg)
                s.send(msg_data_json.encode('utf-8'))


            fstr = self.ID + "_log.txt"
            with open(fstr, 'a') as f:
                f.truncate(0)
                f.close()

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.HOST, self.PORT))
                s.listen(5) # listen for server message

                round = -1        
                while True:
                    round += 1
                    # Receive global model
                    server_data = self.listen_for_broadcast(s)
                    if server_data.get("message", None) == "Training finished":
                        print("Training finished!")
                        return
                    print("I am client " + str(self.ID_num))
                    print("Receiving new global model")
                    server_model = server_data['model']
                    # Test local model
                    if self.mini_batch_GD == 1:
                        local_model = UserAVG(self.ID_num, server_model, lr, batch_size, data)
                    else:
                        local_model = UserAVG(self.ID_num, server_model, lr, None, data)
                    
                    training_loss = local_model.train_loss()
                    test_accuracy = local_model.test()



                    #write to client.txt file
                    with open(fstr, 'a') as f:
                        recordstr = "{}, {}, {}\n".format(str(round), str(training_loss), str(test_accuracy))
                        f.write(recordstr)
                        f.close()
                    
                    # Train local model
                    print("Local training...")
                    local_model.train(E)
                    # Send local model
                    print("Sending new local model")
                    msg = pickle.dumps({"model": local_model.model, "type": "model", "id": self.ID_num})

                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s2.connect((self.HOST, 6000)) #"A", 6000
                        messages = self.split_byte_string(msg)
                        messages.append(b'')
                        
                        for packet in messages:
                            s2.send(packet)

        except Exception as e:
            print("[Client] Can't connect to the Server:  ", str(e))
            
            exit()

 

    def listen_for_broadcast(self, s):
        # Simply listen for a message from the server
        try:
            c, addr = s.accept()
            byte_string = b''
            data = c.recv(1024)
            while len(data) > 0:
                byte_string += data
                data = c.recv(1024)
            server_model = pickle.loads(byte_string) 
            return server_model
            
        except Exception as e:
            print("Failed to open socket as server: " + str(e))

    def split_byte_string(self, byte_string):
        # split byte string into packets of 1024 each
        return [byte_string[i:i+1024] for i in range(0, len(byte_string), 1024)]


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
    


    


client = Client()
client.run()


