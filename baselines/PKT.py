import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PKT(nn.Module):

    def __init__(self, nQues, szRnnIn, szRnnOut, nRnnLayer, szOut, dropout, opt):
        super(PKT, self).__init__()
        self.encoderQuesLabel = nn.Embedding(num_embeddings=2 * nQues + 1, embedding_dim=szRnnIn, padding_idx=0)
        self.rnn = nn.LSTM(input_size=szRnnIn, hidden_size=szRnnOut, num_layers=nRnnLayer, batch_first=True, dropout=dropout)

        self.encoderNextQues = nn.Embedding(num_embeddings=nQues + 1, embedding_dim=szRnnOut, padding_idx=0)

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

    def forward(self, currQuesAddLabel, nextSkill_oneHot, nextQuesID):
        bsz, maxLen = currQuesAddLabel.size()
        embQuesLabel = self.encoderQuesLabel(currQuesAddLabel)
        difficultySkillLabel = self.sigmoid(embQuesLabel)
        rnn_in = embQuesLabel * (1 + difficultySkillLabel)

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