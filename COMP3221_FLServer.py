import sys
import socket
import json
import time

T = 100
N_CLIENTS = 5


class Server:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = "Server"
        self.PORT = int(sys.argv[1]) #(PORT 6000)
        self.sub_client = sys.argv[2] #(sub_client)

        self.clients = {}  #{"client1": <data_size>, "client2": <data_size>...}
        self.initialisation = True

        self.client_models = {}

        # TODO: Init global model
        self.global_iteration = 1
        self.global_model = None

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
                        
                        # initialisation stage
                        if self.initialisation:
                            if handshake_start is None:
                                # Start timer
                                handshake_start = time.time()
                                s.settimeout(30)
                                continue
                            elif time.time() - handshake_start < 30: 
                                s.settimeout(int(30 - (time.time() - handshake_start)))
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
                        if not data:
                            return
                        data_json = json.loads(data.decode('utf-8')) 
                        # Accepting client case, new clients will be applied next round
                        #{"id": 1, "data_size": 99}
                        if data_json["type"] == "handshake" and self.clients.get(data_json["id"], None) == None:
                            self.clients[data_json["id"]] = data_json["data_size"]
                        
                        # Accepting models case
                        #{"id": 1, "model": <model parameters>}
                        if data_json["type"] == "model":
                            self.client_models[data_json["id"]] = data_json["model"]
                            print("Getting local model from client " + str(data_json["id"]))

                        
                    # Received all models for this round
                    print(self.client_models)
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
                    msg = {"model": "model" + str(self.global_iteration)}
                    msg_data_json = json.dumps(msg)
                    s.send(msg_data_json.encode('utf-8'))
                except Exception as e:
                    print("Failed to send model to client " + str(id) + ": " + str(e))

    def aggergate(self):
        print("Aggregating new global model")
        print(self.client_models)
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
                    msg_data_json = json.dumps(msg)
                    s.send(msg_data_json.encode('utf-8'))
                except Exception as e:
                    print("Failed to send model to client " + str(id) + ": " + str(e))



    # def send_parameters(server_model, users):
    #     for user in users:
    #         user.set_parameters(server_model)


    # def aggregate_parameters(server_model, users, total_train_samples):
    #     # Clear global model before aggregation
    #     for param in server_model.parameters():
    #         param.data = torch.zeros_like(param.data)
            
    #     for user in users:
    #         for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
    #             server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    #     return server_model

    # def evaluate(user):
    #     total_accurancy = 0
    #     for user in users:
    #         total_accurancy += user.test()
    #     return total_accurancy/len(users)

if __name__ == '__main__':
    server = Server()
    server.run()