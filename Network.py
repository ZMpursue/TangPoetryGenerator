from DataLoader import *
from Config import *
from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle

class Peom(paddle.nn.Layer):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(Peom, self).__init__()
        self.embeddings = paddle.nn.Embedding(vocab_size,embedding_dim)
        self.lstm = paddle.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=Config.num_layers,
        )
        self.linear = paddle.nn.Linear(in_features=hidden_dim,out_features=vocab_size)
    def forward(self,input_,hidden=None):
        seq_len, batch_size = paddle.shape(input_)
        embeds = self.embeddings(input_)
        if hidden is None:
            output,hidden = self.lstm(embeds)
        else:
            output,hidden = self.lstm(embeds,hidden)
        output = paddle.reshape(output,[seq_len*batch_size,Config.hidden_dim])
        output = self.linear(output)
        return output,hidden
