import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import Counter
from tqdm import tqdm
import pickle
import glob

torch.manual_seed(66)

data = []
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/data/vocab.pkl', 'rb') as f:
    data = pickle.load(f)
    # print(data)
vocab = dict(data)


key_list = list(vocab.keys())
val_list = list(vocab.values())
def itos(a):
  return key_list[a]


def convert_line2idx(line, vocab):
    """ Converts a string into a list of character indices
    Input
    ------------
    line: str. A line worth of data
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    -------------
    line_data: List[int]. List of indices corresponding to the characters
                in the input line.
    """
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data


class CharLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dimension, hidden_shape, n_layers, drop_prob=0.2, lr=0.001):
    super(CharLSTM, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dimension = embedding_dimension
    self.n_layers = n_layers
    self.hidden_shape = hidden_shape
    self.lr = lr
    if n_layers == 1:
      self.drop_prob = 0.0
    else:
      self.drop_prob = drop_prob

    self.embeddings = nn.Embedding(
        num_embeddings= vocab_size,
        embedding_dim= embedding_dimension,
        max_norm= 1,
        )

    self.lstm = nn.LSTM(
        input_size=embedding_dimension,
        hidden_size=hidden_shape,
        num_layers=n_layers,
        batch_first=True,
        dropout=self.drop_prob
        )

    self.dropout = nn.Dropout(drop_prob)

    self.linear1 = nn.Linear(hidden_shape, hidden_shape*2)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_shape*2, vocab_size)


  def forward(self, x, hc):
    # print(x.shape)
    x = self.embeddings(x)   #.squeeze()
    # print("After Embedding")
    # print(x.shape)

    x, (h, c) = self.lstm(x, hc)
    # print(x.shape)
    x = self.dropout(x)

    x = x.reshape(x.size()[0]*x.size()[1], self.hidden_shape)
    # print(x.shape)
    x = self.linear1(x)
    # print(x.shape)
    x = self.linear2(self.relu(x))
    # print(x.shape)
    return x, (h, c)


learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 2
k=500
n_epochs = 5
batch_size = 1
lr = learning_rates[2]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_2_0.0001.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
char_model.to(device)


def generate_string(my_string):
  h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
  c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
  val_h = tuple([h_1, c_1])

  character_list = list(my_string)
  encoded_text = convert_line2idx(character_list, vocab)
  for i in range(0,200):
    encoded_text = encoded_text
    encoded_text_2d = [encoded_text]
    encoded_text_torch = torch.tensor(encoded_text_2d)

    inputs = encoded_text_torch.to(device)
    val_h = tuple([each.data for each in val_h])

    output, val_h = char_model.forward(inputs, val_h)
    softmax_output = torch.nn.functional.softmax(output[-1,:], dim=0)
    sampled_indices = torch.multinomial(softmax_output,num_samples = 1).item()
    encoded_text.append(sampled_indices)

  char_string = ''
  for i in range(len(encoded_text)):
    a = itos(encoded_text[i])
    char_string += a
  return char_string


my_string = "The little boy was"
generated_string = generate_string(my_string)
print("Original String:")
print(my_string)
print("Generated String:")
print(generated_string)


my_string = "Once upon a time in"
generated_string = generate_string(my_string)
print("Original String:")
print(my_string)
print("Generated String:")
print(generated_string)


my_string = "With the target in"
generated_string = generate_string(my_string)
print("Original String:")
print(my_string)
print("Generated String:")
print(generated_string)


my_string = "Capitals are big cities. For example,"
generated_string = generate_string(my_string)
print("Original String:")
print(my_string)
print("Generated String:")
print(generated_string)


my_string = "A cheap alternative to"
generated_string = generate_string(my_string)
print("Original String:")
print(my_string)
print("Generated String:")
print(generated_string)

