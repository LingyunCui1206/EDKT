import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import one_hot
import numpy as np

MAX_CODE_LEN = 128

node_count, path_count = np.load("np_counts.npy")

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, n_skills,device):
        super(DKT, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.n_skills = n_skills
        
        self.emb_dim = 128
        self.embed_nodes = nn.Embedding(node_count+2, self.emb_dim) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, self.emb_dim) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(self.emb_dim*3, self.emb_dim*3)
        self.attention_layer = nn.Linear(self.emb_dim*3,1) 
        self.attention_softmax = nn.Softmax(dim=1)
        
        self.rnn = nn.LSTM(self.input_dim*4, self.hidden_dim, self.layer_dim, batch_first=True)
        self.qa_embedding = nn.Embedding(self.n_skills * 2 + 2, self.input_dim, padding_idx=self.n_skills*2+1)
        self.fc = nn.Linear(self.hidden_dim, self.n_skills)

    def forward(self, x, code):
        
        x = self.qa_embedding(x)
        
        # --------- code ---------
        c2v_input = code[:, :, self.n_question*2:].reshape(code.size(0), code.size(1), MAX_CODE_LEN, 3).long() # (b,l,e,3)
        starting_node_index = c2v_input[:,:,:,0] # (b,l,e,1)
        ending_node_index = c2v_input[:,:,:,2] # (b,l,e,1)
        path_index = c2v_input[:,:,:,1] # (b,l,e,1)
        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,e,1) -> (b,l,e,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,e,1) -> (b,l,e,ne)
        path_embed = self.embed_paths(path_index) # (b,l,e,1) -> (b,l,e,pe)
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed), dim=3) # (b,l,e,2ne+pe)
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,e,2ne+pe)
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,e,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,e,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) # (b,l,2ne+pe)
        
        x = torch.cat((x, code_vectors), dim=2)
        # ------------------------
        
        out,hn = self.rnn(x)
        res = self.fc(out)
        x = torch.sigmoid(res)
        # final = (x * one_hot(cshft.long(), self.n_skills+1)[:,:,:-1]).sum(-1)
        return x