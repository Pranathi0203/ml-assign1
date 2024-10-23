import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt 

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h,dr=0.25):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.dropout = nn.Dropout(dr)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        output, _ = self.rnn(inputs)
        output = output[-1]
        output = self.dropout(output)
        output = self.W(output)
        predicted_vector = self.softmax(output)
        return predicted_vector

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    tst = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test]
    
    return tra, val, tst

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    word_embedding = pickle.load(open('Data_Embedding/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    with open('rnntests.txt', 'a') as f_metrics:
        f_metrics.write("Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss\n")

        while not stopping_condition and epoch < args.epochs:
            random.shuffle(train_data)
            model.train()
            print(f"Training started for epoch {epoch + 1}")
            correct = 0
            total = 0
            minibatch_size = 16
            N = len(train_data)

            loss_total = 0
            loss_count = 0
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    input_words = " ".join(input_words)
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)

                    example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()

            print(f"Training completed for epoch {epoch + 1}")
            train_accuracy = correct / total
            train_loss = loss_total / loss_count
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            print(f"Training accuracy for epoch {epoch + 1}: {train_accuracy}")
            print(f"Training loss for epoch {epoch + 1}: {train_loss}")

            model.eval()
            correct = 0
            total = 0
            val_loss_total = 0
            val_loss_count = 0
            print(f"Validation started for epoch {epoch + 1}")
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                correct += int(predicted_label == gold_label)
                total += 1
                val_loss_total += example_loss.data
                val_loss_count += 1

            validation_accuracy = correct / total
            validation_loss = val_loss_total / val_loss_count
            val_accuracies.append(validation_accuracy)
            val_losses.append(validation_loss)

            print(f"Validation accuracy for epoch {epoch + 1}: {validation_accuracy}")
            print(f"Validation loss for epoch {epoch + 1}: {validation_loss}")

  
            f_metrics.write(f"{epoch + 1},{train_accuracy},{train_loss},{validation_accuracy},{validation_loss}\n")

            if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
                stopping_condition = True
                print("Training stopped to avoid overfitting!")
                print(f"Best validation accuracy: {last_validation_accuracy}")
            else:
                last_validation_accuracy = validation_accuracy
                last_train_accuracy = train_accuracy

            epoch += 1

    # Test Accuracy
    print("========== Testing on Test Data ==========")
    model.eval()
    correct = 0
    total = 0
    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")

    # Plot graphs
    epochs_range = range(1, epoch + 1)

    plt.figure(figsize=(10, 5))

    # Plot 1: Training accuracy and Validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot 2: Training accuracy and Training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.title('Training Accuracy and Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
