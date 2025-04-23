# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-10 00:29:34
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:14:50
import torch
import torch.nn as nn
MAX_CODE_LEN = 100

class c2vRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(c2vRNNModel, self).__init__()

        self.embed_nodes = nn.Embedding(node_count+2, 100) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, 100) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(input_dim+300,input_dim+300)
        self.attention_layer = nn.Linear(input_dim+300,1)
        self.prediction_layer = nn.Linear(input_dim+300,1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(2*input_dim+300,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, evaluating=False):
        
        rnn_first_part = x[:, :, :self.input_dim]
        rnn_attention_part = torch.stack([rnn_first_part]*MAX_CODE_LEN,dim=-2)

        c2v_input = x[:, :, self.input_dim:].reshape(x.size(0), x.size(1), MAX_CODE_LEN, 3).long()

        starting_node_index = c2v_input[:,:,:,0]
        ending_node_index = c2v_input[:,:,:,2]
        path_index = c2v_input[:,:,:,1]

        starting_node_embed = self.embed_nodes(starting_node_index)
        ending_node_embed = self.embed_nodes(ending_node_index)
        path_embed = self.embed_paths(path_index)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part), dim=3)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed)
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed))
        context_weights = self.attention_layer(full_embed_transformed)
        attention_weights = self.attention_softmax(context_weights)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2)
        rnn_input = torch.cat((rnn_first_part,code_vectors), dim=2)
        
        out, hn = self.rnn(rnn_input)
        res = self.sig(self.fc(out))
        return res
