import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import glob
import re
import math
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
torch.manual_seed(66)


#Importing vocabulary
data = []
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/data/vocab.pkl', 'rb') as f:
    data = pickle.load(f)
    # print(data)
vocab = dict(data)
print(vocab)

#Provided Functions for data preprocessing
def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from.

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list

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


def convert_files2idx(files, vocab):
    """ This method iterates over files. In each file, it iterates over
    every line. Every line is then split into characters and the characters are
    converted to their respective unique indices based on the vocab mapping. All
    converted lines are added to a central list containing the mapped data.
    Input
    --------------
    files: List[str]. List of files in a particular split
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    ---------------
    data: List[List[int]]. List of lists where each inner list is a list of character
            indices corresponding to a line in the training split.
    """
    data = []

    for file in files:
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            toks = convert_line2idx(line, vocab)
            data.append(toks)

    return data
    

#Reading Data and saving encoded 
train_files_list = get_files(path="/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/data/train")
test_files_list = get_files(path="/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/data/test")
dev_files_list = get_files(path="/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/data/dev")

train_corpus = convert_files2idx(train_files_list, vocab)
test_corpus = convert_files2idx(test_files_list, vocab)
dev_corpus = convert_files2idx(dev_files_list, vocab)

with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/train.pkl', 'wb') as f:
  pickle.dump(train_corpus, f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/test.pkl', 'wb') as f:
  pickle.dump(test_corpus, f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/dev.pkl', 'wb') as f:
  pickle.dump(dev_corpus, f)


#Loading Encoded Data
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/train.pkl', 'rb') as f:
    train_corpus = pickle.load(f)

with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/test.pkl', 'rb') as f:
    test_corpus = pickle.load(f)

with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/dev.pkl', 'rb') as f:
    dev_corpus = pickle.load(f)
    
    
# #Working with only 0.1% data
# train_corpus = train_corpus[0:int(np.abs(len(train_corpus)*0.001))]
# test_corpus = test_corpus[0:int(np.abs(len(test_corpus)*0.001))]
# dev_corpus = dev_corpus[0:int(np.abs(len(dev_corpus)*0.001))]


#Slide and count for 1D
def sliding_window_counts(arr, window_size):
    subarrays = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(window_size,))
    subarrays = subarrays.reshape(-1, window_size)  # Reshape to 2D

    counter = Counter(map(tuple, subarrays))
    return counter

#Train the dataset by calculating N-grams
def train_n_grams(dataset, n):
  accumulated_counter = Counter()
  
  for i in range(0,len(dataset)):
    seq = [vocab['[PAD]']] * (n-1) + dataset[i]
    seq = np.array(seq)
    window_counts = sliding_window_counts(seq, n)
    accumulated_counter += window_counts
    

  # n_grams_dict = dict(accumulated_counter)
  return accumulated_counter
  

n = 4
all_n_gram = Counter()
all_n_1_gram = Counter()
accumulated_counter = Counter()
xy = np.ceil(len(train_corpus)/10000)
progress_bar = tqdm(total=xy, desc="Progress")
for i in range(0,len(train_corpus),10000):
  temp_n = train_n_grams(train_corpus[i:i+10000], n)
  temp_n1 = train_n_grams(train_corpus[i:i+10000], n-1)

  all_n_gram += temp_n
  all_n_1_gram += temp_n1
  progress_bar.update(1)
progress_bar.close()
all_n_gram = dict(all_n_gram)
all_n_1_gram = dict(all_n_1_gram)


with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/all_n_gram.pkl', 'wb') as f:
  pickle.dump(all_n_gram, f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/all_n_1_gram.pkl', 'wb') as f:
  pickle.dump(all_n_1_gram, f)


with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/all_n_gram.pkl', 'rb') as f:
    all_n_gram = pickle.load(f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/all_n_1_gram.pkl', 'rb') as f:
    all_n_1_gram = pickle.load(f)
    
    
#As conditional probability is equivalent to count of n tuples
print("Number of parameters (Probabilities in train set) = ", len(all_n_gram))


def get_n_gram_count(k):
  if k in all_n_gram:
    return all_n_gram[k]
  else:
    return 0

def get_n_1_gram_count(k):
  if k in all_n_1_gram:
    return all_n_1_gram[k]
  else:
    return 0
    

total_perplexity = 0
n_char = len(vocab)
# all_test_prob = {}
n=4

for i in range(len(test_corpus)):
  # total_words = 0
  seq = [vocab['[PAD]']] * (n-1) + test_corpus[i]
  # seq = test_corpus[i]
  seq = np.array(seq)
  # print(seq.shape[0],n)
  a = np.lib.stride_tricks.sliding_window_view(seq, window_shape=(n,))
  a = a.reshape(-1, n)
  b = a[:,:-1]
  c = list(map(tuple, a))
  d = list(map(tuple, b))
  all_c = np.array(list(map(get_n_gram_count,c)))
  all_d = np.array(list(map(get_n_1_gram_count,d)))
  all_c = all_c + 1
  all_d = all_d + n_char
  prob = all_c /all_d
  log_prob = np.log2(prob)
  # total_log_prob = total_log_prob + np.sum(log_prob)
  # total_words = total_words + log_prob.shape[0]
  seq_loss = (-1 * np.sum(log_prob)) / log_prob.shape[0]
  seq_perplexity = np.exp2(seq_loss)
  total_perplexity += seq_perplexity

# test_loss = (-1 * total_log_prob) / total_words
test_perplexity = total_perplexity/len(test_corpus)
# print("Test Loss = ", test_loss)
print("Test Perplexity = ", test_perplexity)




