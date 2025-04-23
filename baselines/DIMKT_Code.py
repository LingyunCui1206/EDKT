from torch import cat, squeeze, unsqueeze, sum
from torch.nn import Embedding, Module, Sigmoid, Tanh, Dropout, Linear, Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn

import os
import numpy as np
MAX_CODE_LEN = 128
node_count, path_count = np.load("np_counts.npy")

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DIMKT(Module):
    def __init__(self, num_q, num_c, dropout, emb_size, batch_size, difficult_levels):
        super().__init__()
        self.model_name = "dimkt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.difficult_levels = difficult_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.dropout = Dropout(dropout)
        
        # -------------code-------------
        self.emb_dim = 128
        self.embed_nodes = nn.Embedding(node_count+2, self.emb_dim) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, self.emb_dim) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(self.emb_dim*3, self.emb_dim*3)
        self.attention_layer = nn.Linear(self.emb_dim*3,1) 
        self.attention_softmax = nn.Softmax(dim=1)
        self.qa_embed_code_trans = nn.Linear(self.emb_dim*4,self.emb_dim*1) 
        # -----------------------------

        self.knowledge = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True)

        self.q_emb = Embedding(self.num_q + 1, self.emb_size, padding_idx=self.num_q)
        self.c_emb = Embedding(self.num_c + 1, self.emb_size, padding_idx=self.num_c)
        self.sd_emb = Embedding(self.difficult_levels + 100, self.emb_size, padding_idx=0)
        self.qd_emb = Embedding(self.difficult_levels + 100, self.emb_size, padding_idx=0)
        self.a_emb = Embedding(self.num_q*2 + 2, self.emb_size,padding_idx=2)

        self.linear_1 = Linear(4 * self.emb_size, self.emb_size)
        self.linear_2 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_3 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_4 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = Linear(4 * self.emb_size, self.emb_size)
        
        self.linear_final = Linear(self.emb_size, 10)

    def forward(self, code, q, c, sd, qd, a, qshft, cshft, sdshft, qdshft):
        if self.batch_size != len(q):
            self.batch_size = len(q)
        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        sd_emb = self.sd_emb(sd)
        qd_emb = self.qd_emb(qd)
        a_emb = self.a_emb(a)

        target_q = self.q_emb(qshft)
        target_c = self.c_emb(cshft)
        target_sd = self.sd_emb(sdshft)
        target_qd = self.qd_emb(qdshft)
        
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
        
        acode_embed_data = torch.cat((a_emb, code_vectors), dim=2)
        acode_embed_data = self.qa_embed_code_trans(acode_embed_data)
        # ------------------------

        input_data = cat((q_emb, c_emb, sd_emb, qd_emb), -1)
        input_data = self.linear_1(input_data)
        target_data = cat((target_q, target_c, target_sd, target_qd), -1)
        target_data = self.linear_1(target_data)
        
        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        sd_emb = cat((padd, sd_emb), 1)
        slice_sd_embedding = sd_emb.split(1, dim=1)
        shape = list(acode_embed_data.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        acode_embed_data = cat((padd, acode_embed_data), 1)
        slice_a_embedding = acode_embed_data.split(1, dim=1)

        shape = list(input_data.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        input_data = cat((padd, input_data), 1)
        slice_input_data = input_data.split(1, dim=1)
        qd_emb = cat((padd, qd_emb), 1)
        slice_qd_embedding = qd_emb.split(1, dim=1)

        k = self.knowledge.repeat(self.batch_size, 1).cuda()

        h = list()
        # h.append(unsqueeze(k, dim=1))
        # h.append(unsqueeze(k, dim=1))

        seqlen = q.size(1)
        for i in range(1, seqlen + 1):
            sd_1 = squeeze(slice_sd_embedding[i], 1)
            a_1 = squeeze(slice_a_embedding[i], 1)
            qd_1 = squeeze(slice_qd_embedding[i], 1)
            input_data_1 = squeeze(slice_input_data[i], 1)

            qq = input_data_1-k

            gates_SDF = self.linear_2(qq)
            gates_SDF = self.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = self.tanh(SDFt)
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF * SDFt

            x = cat((SDFt, a_1), -1)
            gates_PKA = self.linear_4(x)
            gates_PKA = self.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = self.tanh(PKAt)

            PKAt = gates_PKA * PKAt

            ins = cat((k, a_1, sd_1, qd_1), -1)
            gates_KSU = self.linear_6(ins)
            gates_KSU = self.sigmoid(gates_KSU)

            k = gates_KSU * k + (1 - gates_KSU) * PKAt
            
            logits_i = self.linear_final(k)
            h_i = unsqueeze(logits_i, dim=1)
            h.append(h_i)
            
        # print('out')
        output = cat(h, axis=1)
        # print(output.size())

        y = self.sigmoid(output)
        return y