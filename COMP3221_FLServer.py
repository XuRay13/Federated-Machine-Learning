import sys
import socket
import json
import time

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


T = 100
N_CLIENTS = 5

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
    

class Server:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = "Server"
        self.PORT = int(sys.argv[1]) #(PORT 6000)
        self.sub_client = sys.argv[2] #(sub_client)

        self.clients = {}  #{"client1": <data_size>, "client2": <data_size>...}
        self.initialisation = True

        self.client_models = {}

        self.global_iteration = 1
        self.global_model = MCLR()
        self.total_train_size = 0

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # print("Server host names: ", self.ID, " Port: ", self.PORT)
                s.bind((self.HOST, self.PORT))
                s.listen(N_CLIENTS)
                
                handshake_start = None
                try:
                    while True:
                        c, addr = s.accept()
                        data = c.recv(1024)
                        if not data:
                            return
                        data_json = json.loads(data.decode('utf-8')) 

                        # Accepting client
                        #{"id": "client1", "data_size": 99}
                        if data_json["type"] == "handshake" and self.clients.get(data_json["id"], None) == None:
                            self.clients[data_json["id"]] = data_json["data_size"]
                            self.total_train_size += data_json["data_size"]
                        
                        # initialisation stage
                        if self.initialisation:
                            if handshake_start is None:
                                # Start timer
                                handshake_start = time.time()
                                s.settimeout(10)
                                continue
                            elif time.time() - handshake_start < 10: 
                                s.settimeout(int(10 - (time.time() - handshake_start)))
                                # Check if 30 seconds has passed
                                continue
                            else:
                                # 30 seconds has passed, first broadcast
                                self.broadcast()
                                self.initialisation = False
                                break
                except Exception as e:
                    # 30 seconds has passed, first broadcast
                    self.broadcast()
                    self.initialisation = False
                
                s.settimeout(None)
                    
                # Global communication rounds stage
                while self.global_iteration <= T:
                    
                    print("Global Iteration " + str(self.global_iteration))

                    n_clients = len(self.clients)
                    print("Total Number of clients: " + str(n_clients))
                    while len(self.client_models) < n_clients:
                        c, addr = s.accept()
                        data = c.recv(1024)
                        byte_string = b''
                        while len(data) > 0:
                            byte_string += data
                            data = c.recv(1024)
                        data_json = pickle.loads(byte_string)
                        # data_json = json.loads(data.decode('utf-8')) 
                        # Accepting client case, new clients will be applied next round
                        #{"id": 1, "data_size": 99}
                        if data_json["type"] == "handshake" and self.clients.get(data_json["id"], None) == None:
                            self.clients[data_json["id"]] = data_json["data_size"]
                            self.total_train_size += data_json["data_size"]
                        
                        # Accepting models case
                        #{"id": 1, "model": <model parameters>}
                        if data_json["type"] == "model":
                            self.client_models[data_json["id"]] = data_json["model"]
                            print("Getting local model from client " + str(data_json["id"]))

                        
                    # Received all models for this round
                    self.aggergate()
                    self.broadcast()
                    self.global_iteration += 1


                self.broadcast_finished()

        except Exception as e:
            print("[Server] Error: " + str(e))

    def broadcast(self):
        print("Broadcasting new global model")
        for id in self.clients.keys():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((self.HOST, self.PORT+id))
                    # Send global model
                    msg_string = pickle.dumps({"model": self.global_model})
                    messages = self.split_byte_string(msg_string)
                    messages.append(b'')
                    # print(len(msg_string), len(messages))
                    # msg_data_json = json.dumps(msg)
                    for packet in messages:
                        s.send(packet)

                except Exception as e:
                    print("Failed to send model to client " + str(id) + ": " + str(e))

    def split_byte_string(self, byte_string):
        # split byte string into packets of 1024 each
        return [byte_string[i:i+1024] for i in range(0, len(byte_string), 1024)]

    def aggergate(self):
        print("Aggregating new global model")
        # print(self.client_models)
        # Clear global model before aggregation
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for id, user in self.client_models.items():
            for server_param, user_param in zip(self.global_model.parameters(), user.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * self.clients[id] / self.total_train_size
        # Reset client models after aggregation
        self.client_models = {}


    def broadcast_finished(self):
        print("Model finished training!")
        for id in self.clients.keys():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((self.HOST, self.PORT+id))
                    # Send global model
                    msg = {"model": None, "message": "Training finished"}
                    msg_data_json = pickle.dumps(msg)
                    s.send(msg_data_json.encode('utf-8'))
                except Exception as e:
                    print("Failed to send model to client " + str(id) + ": " + str(e))



if __name__ == '__main__':
    server = Server()
    server.run()

