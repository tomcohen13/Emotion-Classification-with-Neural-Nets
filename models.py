
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE    = 128

class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embedding = nn.Embedding(embeddings.shape[0], EMBEDDING_DIM)
        self.embedding.weight = nn.Parameter(embeddings.type(torch.FloatTensor), requires_grad = False)
        self.l1     = nn.Linear(EMBEDDING_DIM, 100)
        self.relu   = nn.ReLU()
        self.l2     = nn.Linear(100, 4)
        self.sm     = nn.Softmax()     
        
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        x = self.embedding(x)
        x = torch.sum(x, dim = 1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        out = self.sm(x)
        return out
            
class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings, hidden, layers = 2):
        super(RecurrentNetwork, self).__init__()

        # TODO: Here, create any layers and attributes your network needs.
        self.layers = layers
        self.hidden = hidden
        self.embedding = nn.Embedding.from_pretrained(embeddings.float())
        self.rnn    = nn.LSTM(input_size = EMBEDDING_DIM, hidden_size = hidden, num_layers= 2, bidirectional=True)
        self.lin    = nn.Linear(hidden, 4)
        self.f      = nn.ReLU()

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class

        x = self.embedding(x)
        lens = torch.zeros(x.shape[0])
        
        for i in range(x.shape[0]):
            j = 0
            while j < x.shape[1] and x[i][j][0] is not 0:
                j += 1 
            lens[i] = j

        x = rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        
        # Forward pass
        x, (ht, ct) = self.rnn(x)

        # return the last hidden state of the RNN
        x = self.lin(ht[-1])

        return x
        
        
# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        raise NotImplementedError
