#Importing all the libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import glob

torch.manual_seed(66)


#Printing the dictionary
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
  

#Retriving the saved files 
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/train.pkl', 'rb') as f:
    train_corpus = pickle.load(f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/test.pkl', 'rb') as f:
    test_corpus = pickle.load(f)
with open('/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/mp3_release/all_corpus/dev.pkl', 'rb') as f:
    dev_corpus = pickle.load(f)
    

#Padding after each sequence
k = 500

for i in range(len(train_corpus)):
  m = len(train_corpus[i]) % k
  if m != 0:
    for j in range(k-m):
      train_corpus[i].append(vocab['[PAD]'])
  train_corpus[i].append(vocab['[PAD]'])

for i in range(len(test_corpus)):
  m = len(test_corpus[i]) % k
  if m != 0:
    for j in range(k-m):
      test_corpus[i].append(vocab['[PAD]'])
  test_corpus[i].append(vocab['[PAD]'])

for i in range(len(dev_corpus)):
  m = len(dev_corpus[i]) % k
  if m != 0:
    for j in range(k-m):
      dev_corpus[i].append(vocab['[PAD]'])
  dev_corpus[i].append(vocab['[PAD]'])


# #Taking 5% data
# train_corpus = train_corpus[0:int(np.abs(len(train_corpus)*0.05))]
# test_corpus = test_corpus[0:int(np.abs(len(test_corpus)*0.05))]
# dev_corpus = dev_corpus[0:int(np.abs(len(dev_corpus)*0.05))]


#Making train, dev and test sets
trainX = []
trainY = []
testX = []
testY = []
devX = []
devY = []
for i in range(0, len(train_corpus)):
  for j in range(0, len(train_corpus[i]) - 1, k):
    seq_in = train_corpus[i][j : j+k]
    if(len(seq_in)<500):
      print(i,j,len(seq_out))
    seq_out = train_corpus[i][j+1 : j+k+1]
    if(len(seq_out)<500):
      print(i,j,len(seq_out))
    trainX.append(seq_in)
    trainY.append(seq_out)
train_corpus = None

for i in range(0, len(dev_corpus)):
  for j in range(0, len(dev_corpus[i]) - 1, k):
    seq_in = dev_corpus[i][j : j+k]
    seq_out = dev_corpus[i][j+1 : j+k+1]
    devX.append(seq_in)
    devY.append(seq_out)
dev_corpus = None

for i in range(0, len(test_corpus)):
  for j in range(0, len(test_corpus[i]) - 1, k):
    seq_in = test_corpus[i][j : j+k]
    seq_out = test_corpus[i][j+1 : j+k+1]
    testX.append(seq_in)
    testY.append(seq_out)
test_corpus = None


#Reshaping
trainX = torch.tensor(trainX).reshape(len(trainX), k)
trainY = torch.tensor(trainY).reshape(len(trainY), k)
testX = torch.tensor(testX).reshape(len(testX), k)
testY = torch.tensor(testY).reshape(len(testY), k)
devX = torch.tensor(devX).reshape(len(devX), k)
devY = torch.tensor(devY).reshape(len(devY), k)


#Keeping till batch_size
batch_size = 32
n_batches = trainX.shape[0]//batch_size
trainX = trainX[:n_batches * batch_size]
trainY = trainY[:n_batches * batch_size]

n_batches = devX.shape[0]//batch_size
devX = devX[:n_batches * batch_size]
devY = devY[:n_batches * batch_size]

n_batches = testX.shape[0]//batch_size
testX = testX[:n_batches * batch_size]
testY = testY[:n_batches * batch_size]

#calculating Weights
vocab_size = len(vocab)
bin_tens = np.apply_along_axis(lambda x: np.bincount(x, minlength=vocab_size), axis=1, arr=trainX.numpy())
bin_tens = np.sum(bin_tens, axis=0)
# print(bin_tens.shape)
bin_tens = bin_tens + 1
# print(bin_tens)
weights = torch.tensor(bin_tens/sum(bin_tens))
print(weights.shape)

#I built it misunderstanding the question. This is basically of no-use
def one_hot_tensor(arr, vocab_len):
  one_hot = torch.zeros(arr.shape[0]* arr.shape[1], vocab_len)
  one_hot[torch.arange(one_hot.shape[0]), arr.flatten()] = 1
  one_hot = one_hot.reshape((arr.shape[0], arr.shape[1], vocab_len))
  return one_hot
  
  
#Defining device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Tranfering weight to cuda() from cpu()
weights = weights.float().to(device)

#Defining Class
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
    
    
#Train Function  
def train(model, train_loader, dev_loader, n_layers, hidden_shape, embedding_dimension, batch_size = 32, n_seqs = 500, epochs=2, lr=0.001, clip=5):


  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')

  counter = 0
  best_loss = np.inf
  # n_chars = len(net.chars)
  train_per_all = np.zeros(epochs)
  dev_per_all = np.zeros(epochs)
  for e in range(epochs):
    h_0 = torch.zeros(n_layers, batch_size, hidden_shape).cuda().requires_grad_(True)
    c_0 = torch.zeros(n_layers, batch_size, hidden_shape).cuda().requires_grad_(True)
    h = tuple([h_0, c_0])
    train_losses = []
    model.train()
    for inputs, targets in train_loader:

      counter += 1
      inputs, targets = inputs.to(device), targets.to(device)
      h = tuple([each.data for each in h])
      model.zero_grad()

      output, h = model.forward(inputs, h)
      loss = criterion(output, targets.view(batch_size * n_seqs).type(torch.cuda.LongTensor))
      # print(loss.item())
      loss.sum().backward()
      train_losses.extend(loss.tolist())

      nn.utils.clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
    train_loss = np.mean(train_losses)
    train_perplexity = np.exp2(train_loss)
    train_per_all[e] = train_perplexity
    # print('Epoch = ', e, ' Train Loss = ',train_loss)

    h_1 = torch.zeros(n_layers, batch_size, hidden_shape).cuda()
    c_1 = torch.zeros(n_layers, batch_size, hidden_shape).cuda()
    val_h = tuple([h_1, c_1])
    # model.eval()
    with torch.no_grad():
      model.eval()
      val_losses = []
      # print("here")
      for inputs, targets in dev_loader:
        # print("Here")
        # counter += 1
        inputs, targets = inputs.to(device), targets.to(device)
        val_h = tuple([each.data for each in val_h])

        output, val_h = model.forward(inputs, val_h)
        val_loss = criterion(output, targets.view(batch_size * n_seqs).type(torch.cuda.LongTensor))
        # print(val_loss.item())
        val_losses.extend(val_loss.tolist())
      validation_loss = np.mean(val_losses)
      val_perplexity = np.exp2(validation_loss)
      dev_per_all[e] = val_perplexity
      if best_loss > val_perplexity:
        best_loss = val_perplexity
        torch.save({
            "model_param": model.state_dict(),
            "optim_param": optimizer.state_dict(),
            "lowest_dev_perplexity": best_loss,
            "epoch": e+1,
            "learning_rate": lr},
                 f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{n_layers}_{lr}.pth")
      print('Epoch = ', e+1,'/',epochs,'Train Perplexity = ',train_perplexity, ' Validation Perplexity = ',val_perplexity)
  
  print("Best dev perplexity = ", best_loss)
  return train_per_all, dev_per_all, model, best_loss
  
  
  
#Layer-1, LR = 0.0001 
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 1
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[0]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_{hidden_layers}_{lr}.png")



model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)


#Layer-2, LR = 0.0001
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 2
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[0]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_2_0.0001.png")

model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)


#Layer-1, LR = 0.00001
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 1
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[1]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_1_0.00001.png")

model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)


#Layer-2, LR = 0.00001
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 2
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[1]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_2_0.00001.png")


model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)



#Layer-1, LR = 0.000001
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 1
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[2]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_1_0.000001.png")


model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)

#Layer-2, LR = 0.000001
learning_rates = np.array([0.0001, 0.00001, 0.000001])
embedding_dimension = 50
hidden_shape = 200
hidden_layers = 2
k=500
n_epochs = 5
batch_size = 32
lr = learning_rates[2]

print("For Hidden Layers = ", hidden_layers," and Learning Rate = ",lr)

char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
char_model.to(device)
print(char_model)

train_dataloader = DataLoader(list(zip(trainX,trainY)), shuffle=False, batch_size=batch_size)
dev_dataloader = DataLoader(list(zip(devX,devY)), shuffle=False, batch_size=batch_size)

# print(type(train_dataloader))
# print(type(dev_dataloader))

train_per, dev_per, model, min_per = train(model= char_model, train_loader= train_dataloader, dev_loader= dev_dataloader,
      n_layers= hidden_layers, hidden_shape= hidden_shape, embedding_dimension= embedding_dimension,
      batch_size= batch_size, n_seqs= k, epochs= n_epochs, lr = lr)

plt.plot(train_per, label = 'train_perplexity')
plt.plot(dev_per, label = 'dev_perplexity')
plt.legend()
plt.show()
plt.savefig("/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/train_dev_2_0.000001.png")


model_path = f"/uufs/chpc.utah.edu/common/home/u1472438/Assignment3/best_model_{hidden_layers}_{lr}.pth"
checkpoint = torch.load(model_path)
char_model = CharLSTM(vocab_size= len(vocab), embedding_dimension= embedding_dimension,
                      hidden_shape=hidden_shape, n_layers=hidden_layers, drop_prob= 0.2, lr= lr)
optimizer = torch.optim.Adam(char_model.parameters(), checkpoint["learning_rate"])
char_model.load_state_dict(checkpoint["model_param"])
optimizer.load_state_dict(checkpoint["optim_param"])
num_param = sum ( p.numel () for p in char_model.parameters () )
print("Lowest Dev Perplexity = ", checkpoint["lowest_dev_perplexity"])
print("Learning Rate = ", checkpoint["learning_rate"])
print("Epoch = ", checkpoint["epoch"])
print("Parameters = ",num_param)



char_model.to(device)
test_dataloader = DataLoader(list(zip(testX,testY)), shuffle=False, batch_size=batch_size)
test_losses = []
criterion = nn.CrossEntropyLoss(weight = weights,ignore_index=vocab['[PAD]'], reduction= 'none')
# print("here")
h_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
c_1 = torch.zeros(hidden_layers, batch_size, hidden_shape).cuda()
val_h = tuple([h_1, c_1])
for inputs, targets in test_dataloader:
  # print("Here")
  # counter += 1
  inputs, targets = inputs.to(device), targets.to(device)
  val_h = tuple([each.data for each in val_h])

  output, val_h = char_model.forward(inputs, val_h)
  val_loss = criterion(output, targets.view(batch_size * k).type(torch.cuda.LongTensor))
  # print(val_loss.item())
  test_losses.extend(val_loss.tolist())
test_loss = np.mean(test_losses)
test_perplexity = np.exp2(test_loss)
print("Test Perplexity = ",test_perplexity)

