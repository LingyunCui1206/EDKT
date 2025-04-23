import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

import numpy as np
MAX_CODE_LEN = 128
node_count, path_count = np.load("np_counts.npy")


class DKVMN(Module):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        
        # -------------code-------------
        self.emb_dim = 128
        self.embed_nodes = nn.Embedding(node_count+2, self.emb_dim) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, self.emb_dim) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(self.emb_dim*3, self.emb_dim*3)
        self.attention_layer = nn.Linear(self.emb_dim*3,1) 
        self.attention_softmax = nn.Softmax(dim=1)
        # -----------------------------

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c+1, self.dim_s,padding_idx=self.num_c)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2 + 2, self.dim_s,padding_idx=self.num_c * 2 + 1)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 10)

        self.e_layer = Linear(self.dim_s*4, self.dim_s)
        self.a_layer = Linear(self.dim_s*4, self.dim_s)

    def forward(self, code, q, r, qtest=False):
        batch_size = q.shape[0]
        k = self.k_emb_layer(q)
        v = self.v_emb_layer(r)
        
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
        
        v = torch.cat((v, code_vectors), dim=2)
        # ------------------------

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        
        # print(f"p: {p.shape}")
        # p = p.squeeze(-1)
        return p

