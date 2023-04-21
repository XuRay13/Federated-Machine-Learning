import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

import sys


def show_accuracy_graph(avg_acc):
        plt.figure(1,figsize=(5, 5))
        plt.plot(avg_acc, label="FedAvg", linewidth  = 1)
        plt.ylim([0,  0.99])
        plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
        plt.ylabel('Testing Acc')
        plt.xlabel('Global rounds')
        plt.show()

def show_training_loss_graph(avg_loss):
    plt.figure(1,figsize=(5, 5))
    plt.plot(avg_loss, label="FedAvg", linewidth  = 1)
    plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.show()


total_loss = []
total_acc = []

fstr = sys.argv[1] + "_log.txt"
with open(fstr, 'r') as f:
      
      while True:
            line = f.readline().strip("\n")
            if line == "":
                  break
            split = line.split(", ")
            total_loss.append(float(split[1]))
            total_acc.append(float(split[2]))

f.close()

# avg_loss = total_loss / 100
# avg_acc = total_acc / 100
# print(avg_loss, avg_acc)
show_accuracy_graph(total_acc)
show_training_loss_graph(total_loss)