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





class Client:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = sys.argv[1] #("client1")
        self.PORT = int(sys.argv[2]) #(PORT 6000)

    def run(self):
        X_train, y_train, X_test, y_test, train_samples, test_samples = self.get_data()
        
        while(True):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:#s = socket.socket()         # Create a socket object
                    s.connect((self.HOST, 6000)) #"A", 6000
                    print("established connection to server")

                    msg = {"id": self.ID, "data_size": len(X_train)}

                    msg_data_json = json.dumps(msg)
                    s.send(msg_data_json.encode('utf-8'))
                    print("[Client{}] sent data...".format(self.ID))
                    
                    
            except Exception as e:
                print("[Client] Can't connect to the Server:  ", str(e))
                exit()






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


