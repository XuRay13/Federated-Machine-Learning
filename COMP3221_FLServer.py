import sys
import socket
import json
import time



class Server:
    def __init__(self, *args):
        self.HOST = 'localhost'

        self.ID = "Server"
        self.PORT = int(sys.argv[1]) #(PORT 6000)
        self.sub_client = sys.argv[2] #(sub_client)

        self.clients = {}  #{"client1": <data_size>, "client2": <data_size>...}

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

server = Server()
server.run()