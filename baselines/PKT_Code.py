import torch
import torch.nn as nn

import numpy as np
MAX_CODE_LEN = 128
node_count, path_count = np.load("np_counts.npy")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PKT(nn.Module):

    def __init__(self, nQues, szRnnIn, szRnnOut, nRnnLayer, szOut, dropout, opt):
        super(PKT, self).__init__()
        self.nQues = nQues
        self.encoderQuesLabel = nn.Embedding(num_embeddings=2 * nQues + 1, embedding_dim=szRnnIn, padding_idx=0)
        self.rnn = nn.LSTM(input_size=szRnnIn*4, hidden_size=szRnnOut, num_layers=nRnnLayer, batch_first=True, dropout=dropout)

        self.encoderNextQues = nn.Embedding(num_embeddings=nQues + 1, embedding_dim=szRnnOut, padding_idx=0)
        
        # -------------code-------------
        self.emb_dim = 128
        self.embed_nodes = nn.Embedding(node_count+2, self.emb_dim) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, self.emb_dim) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(self.emb_dim*3, self.emb_dim*3)
        self.attention_layer = nn.Linear(self.emb_dim*3,1) 
        self.attention_softmax = nn.Softmax(dim=1)
        # -----------------------------

        self.transL = nn.Linear(2 * szRnnOut, szOut)
        self.transDiff = nn.Linear(szRnnOut, szOut)
        self.transAlpha = nn.Linear(szRnnOut, szOut)
        self.transK = nn.Linear(2 * szRnnOut, szOut)
        self.transG = nn.Linear(2 * szRnnOut, szOut)
        self.transS = nn.Linear(2 * szRnnOut, szOut)

        self.sigmoid = nn.Sigmoid()
        self.opt = opt

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(self.opt.n_rnn_layer, bsz, self.opt.sz_rnn_out).zero_(),
                weight.new(self.opt.n_rnn_layer, bsz, self.opt.sz_rnn_out).zero_())

    def forward(self, code, currQuesAddLabel, nextQuesID):
        bsz, maxLen = currQuesAddLabel.size()
        embQuesLabel = self.encoderQuesLabel(currQuesAddLabel)
        difficultySkillLabel = self.sigmoid(embQuesLabel)
        rnn_in = embQuesLabel * (1 + difficultySkillLabel)
        
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
        
        rnn_in = torch.cat((rnn_in, code_vectors), dim=2)
        # ------------------------

        self.hidden_state = self.init_hidden(bsz)
        rnn_output, self.hidden_state = self.rnn(rnn_in, self.hidden_state)

        embNextQues = self.encoderNextQues(nextQuesID)
        difficultyNextSkill = self.sigmoid(embNextQues)
        nextInput = embNextQues * (1 + difficultyNextSkill)
        nextFullInfo = torch.cat([rnn_output, nextInput], dim=2)

        L_skill = self.sigmoid(self.transL(nextFullInfo))
        Diff = self.sigmoid(self.transDiff(embNextQues))
        q_alpha = self.sigmoid(self.transAlpha(embNextQues))
        G = self.sigmoid(self.transG(nextFullInfo))
        S = self.sigmoid(self.transS(nextFullInfo))

        x = 4 * q_alpha * (L_skill -  Diff)
        L = torch.exp(x) / (1 + torch.exp(x))

        c1 = L * (1 - S)
        c2 = (1 - L) * G
        
        predictAllSkill = (c1 + c2).to(device)

#         predict = torch.sum(predictAllSkill * nextSkill_oneHot, dim=2).to(device)

        return predictAllSkill, [L_skill, G, S]