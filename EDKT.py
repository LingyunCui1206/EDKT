# -*- coding: utf-8 -*-
# @Author: lingyunCui
# @Date:   2024-11-04 21:12:20
# @Last Modified by:   lingyunCui
# @Last Modified time: 
import torch
import torch.nn as nn
import os

MAX_CODE_LEN = 128

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class EDKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, my_q_len, my_errT_len, my_rank_len, max_err_Num, err_threshold_emb, device):
        
        super(EDKT, self).__init__()
        
        self.decay_rate = 0.9
        
        self.my_q_len = my_q_len
        self.my_errT_len = my_errT_len
        self.max_err_Num = max_err_Num
        
        self.err_threshold_emb = err_threshold_emb
        
        self.emb_dim = hidden_dim

        self.embed_nodes = nn.Embedding(node_count+2, self.emb_dim) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, self.emb_dim) # adding unk and end
        self.my_emb_qa = nn.Embedding(2 * my_q_len+1, self.emb_dim) # addind pad
        self.my_emb_q = nn.Embedding(my_q_len+1, self.emb_dim) # addind pad
        self.my_emb_errT = nn.Embedding(my_errT_len+1, self.emb_dim) # addind pad
        self.my_emb_errNum = nn.Embedding(100, self.emb_dim)
        self.my_emb_rank = nn.Embedding(my_rank_len+1, self.emb_dim) # addind pad
        self.my_emb_diff = nn.Embedding(150, self.emb_dim) # addind pad
        
        self.my_emb_qa_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.my_emb_q_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.my_emb_errT_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.my_emb_rank_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.my_emb_diff_layer = nn.Linear(self.emb_dim, self.emb_dim)
        
        self.diff_tran_layer = nn.Linear(self.emb_dim*3, self.emb_dim)
        
        self.delta_sig = nn.Sigmoid()
        
        
        self.intra_layer1 = nn.Linear(self.emb_dim*5,self.emb_dim)
        self.intra_tanh1 = nn.Tanh()
        self.intra_layer2 = nn.Linear(self.emb_dim*5,self.emb_dim)
        self.intra_sig1 = nn.Sigmoid()
        
        self.intra_layer3 = nn.Linear(self.emb_dim*5,self.emb_dim)
        self.intra_tanh2 = nn.Tanh()
        self.intra_layer4 = nn.Linear(self.emb_dim*5,self.emb_dim)
        self.intra_sig2 = nn.Sigmoid()
        
        
        self.intra_alpha1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.intra_alpha2 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(self.emb_dim*3, self.emb_dim*3)
        self.attention_layer = nn.Linear(self.emb_dim*3,1)
        self.attention_softmax = nn.Softmax(dim=1)
        

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.emb_dim*4,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        
        
        self.alpha_param1 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.alpha_param2 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.alpha_param3 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.alpha_param4 = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.diff_subj_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.diff_subj_tanh = nn.Tanh()
        self.diff_subj_layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.diff_subj_sig = nn.Sigmoid()
        
        
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device
        
        
        
    
    

    def forward(self, x, my_x_qa, my_x_q, my_x_err, my_x_rank, my_x_diff, evaluating=False):
        
        batch_size, seq_length = my_x_q.size()
        
        my_x_qa_tensor = torch.tensor(my_x_qa).to(self.device)
        my_x_q_tensor = torch.tensor(my_x_q).to(self.device)
        my_x_rank_tensor = torch.tensor(my_x_rank).to(self.device)
        my_x_diff_tensor = torch.tensor(my_x_diff).to(self.device)
        
        my_x_err_tensor = torch.tensor(my_x_err).to(torch.int64).to(self.device)
        
        err_threshold = torch.full((batch_size, self.my_errT_len), self.err_threshold_emb).to(torch.int64).to(self.device)

        
        qa_emb = self.my_emb_qa_layer(self.my_emb_qa(my_x_qa_tensor).to(self.device))
        q_emb = self.my_emb_q_layer(self.my_emb_q(my_x_q_tensor).to(self.device))
        rank_emb = self.my_emb_rank_layer(self.my_emb_rank(my_x_rank_tensor).to(self.device))
        diff_emb = self.my_emb_diff_layer(self.my_emb_diff(my_x_diff_tensor).to(self.device))
        
        err_emb = self.my_emb_errT_layer(self.my_emb_errNum(my_x_err_tensor).to(self.device))
        err_threshold_emb = self.my_emb_errT_layer(self.my_emb_errNum(err_threshold).to(self.device))

        x_input = x[:, :, self.input_dim:].reshape(x.size(0), x.size(1), MAX_CODE_LEN, 3).long()
        
        starting_node_index = x_input[:,:,:,0]
        ending_node_index = x_input[:,:,:,2]
        path_index = x_input[:,:,:,1]
        
        starting_node_embed = self.embed_nodes(starting_node_index)
        ending_node_embed = self.embed_nodes(ending_node_index)
        path_embed = self.embed_paths(path_index)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed), dim=3)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed)
            
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed))
        context_weights = self.attention_layer(full_embed_transformed)
        attention_weights = self.attention_softmax(context_weights)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2)
        
        
        err_count_pooled_list = []
        
        err_emb_all = torch.zeros(batch_size, self.my_errT_len, self.hidden_dim).to(self.device)
        for i in range(seq_length):
            err_emb_all += err_emb[:,i,:,:]
            err_count_delta = err_emb_all - err_threshold_emb
            err_count_delta = self.delta_sig(err_count_delta)
            err_count_pooled = torch.mean(err_count_delta, dim=1, keepdim=True)
            
            err_count_pooled_list.append(err_count_pooled)
        
        err_count_pooled_all = torch.cat(err_count_pooled_list, dim=1)
            
        complie_err_input = torch.cat((qa_emb, code_vectors, err_count_pooled_all), dim=2)
        complie_err = self.intra_tanh1(self.intra_layer1(complie_err_input))
        complie_err_gate = self.intra_sig1(self.intra_layer2(complie_err_input))
        complie_err_gain = complie_err*complie_err_gate
        
        run_err_input = torch.cat((qa_emb, code_vectors, rank_emb), dim=2)
        run_err = self.intra_tanh2(self.intra_layer3(run_err_input))
        run_err_gate = self.intra_sig2(self.intra_layer4(run_err_input))
        run_err_gain = run_err*run_err_gate
        
        intra_err = self.intra_alpha1*complie_err_gain + self.intra_alpha2*run_err_gain
        
        rnn_input = torch.cat((qa_emb, code_vectors), dim=2)
        
        h_out, hn = self.rnn(rnn_input)
        
        err_h = self.alpha_param1*h_out+self.alpha_param2*intra_err
        
        delta_h_out = torch.zeros_like(err_h)
        for i in range(err_h.size(1)-1):
            delta_h = err_h[:,i+1,:]-err_h[:,i,:]
            delta_h_out[:,i+1,:] = delta_h
        
            
        diff_x = torch.cat([q_emb, diff_emb, delta_h_out], dim=2)
        diff_input = self.diff_tran_layer(diff_x)
        
        sub_diff_h = self.diff_subj_tanh(self.diff_subj_layer1(diff_input - err_h))
        sub_diff_h_choose = self.diff_subj_sig(self.diff_subj_layer2(diff_input - err_h))
        sub_acq = sub_diff_h * sub_diff_h_choose
        
        sub_h = self.alpha_param3*err_h+self.alpha_param4*sub_acq
        
        res = self.sig(self.fc(sub_h))
        
        return res, h_out, err_h, sub_h, qa_emb, code_vectors, rnn_input, diff_emb, delta_h_out, complie_err_gain, run_err_gain
    
