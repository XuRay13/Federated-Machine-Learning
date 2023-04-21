import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


def show_accuracy_graph(avg_acc):
        plt.figure(1,figsize=(5, 5))
        plt.plot(avg_acc, label="FedAvg", linewidth  = 1)
        plt.ylim([0,  0.99])
        plt.yticks(np.arange(0, 0.99 + 0.05, 0.05))
      #   plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
        plt.ylabel('Testing Acc')
        plt.xlabel('Global rounds')
        plt.show()

def show_training_loss_graph(avg_loss):
    plt.figure(1,figsize=(5, 5))
    plt.plot(avg_loss, label="FedAvg", linewidth  = 1)
#     plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.show()


total_loss = []
total_acc = []

fstr = ["client1_log.txt", "client2_log.txt", "client3_log.txt", "client4_log.txt", "client5_log.txt"]
f1 = open(fstr[0], 'r')
f2 = open(fstr[1], 'r')
f3 = open(fstr[2], 'r')
f4 = open(fstr[3], 'r')
f5 = open(fstr[4], 'r')

while True:
      line1 = f1.readline().strip("\n")
      line2 = f2.readline().strip("\n")
      line3 = f3.readline().strip("\n")
      line4 = f4.readline().strip("\n")
      line5 = f5.readline().strip("\n")
      if line1 == "":
            break
      
      split1 = line1.split(", ")
      split2 = line2.split(", ")
      split3 = line3.split(", ")
      split4 = line4.split(", ")
      split5 = line5.split(", ")
      
      total_loss.append((float(split1[1]) + float(split2[1]) + float(split3[1]) + float(split4[1]) + float(split5[1])) / 5)
      total_acc.append((float(split1[2]) + float(split2[2]) + float(split3[2]) + float(split4[2]) + float(split5[2])) / 5)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

# avg_loss = total_loss / 100
# avg_acc = total_acc / 100
# print(avg_loss, avg_acc)
show_accuracy_graph(total_acc)
show_training_loss_graph(total_loss)