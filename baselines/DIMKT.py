from torch import cat, squeeze, unsqueeze, sum
from torch.nn import Embedding, Module, Sigmoid, Tanh, Dropout, Linear, Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn

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

    def forward(self, q, c, sd, qd, a, qshft, cshft, sdshft, qdshft):
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

        input_data = cat((q_emb, c_emb, sd_emb, qd_emb), -1)
        input_data = self.linear_1(input_data)
        target_data = cat((target_q, target_c, target_sd, target_qd), -1)
        target_data = self.linear_1(target_data)
        shape = list(sd_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        sd_emb = cat((padd, sd_emb), 1)
        slice_sd_embedding = sd_emb.split(1, dim=1)
        shape = list(a_emb.shape)
        padd = torch.zeros(shape[0], 1, shape[2], device=self.device)
        a_emb = cat((padd, a_emb), 1)
        slice_a_embedding = a_emb.split(1, dim=1)

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