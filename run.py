
import os
import random
import torch

import torch.optim as optim
import numpy as np

from dataloader import get_data_loader
import evaluation
import warnings
warnings.filterwarnings("ignore")

from EDKT import EDKT
from config import Config

def setup_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    
    config = Config()

    setup_seed(3407)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []

    train_loader, test_loader, my_train_qa_loader, my_test_qa_loader, my_train_q_loader, my_test_q_loader, my_q_len, my_train_mainerr_loader, my_test_mainerr_loader, my_train_err_loader, my_test_err_loader, my_train_rank_loader,my_test_rank_loader, max_err_Num, my_train_diff_loader, my_test_diff_loader= get_data_loader(config.bs, config.questions, config.length) # 128 10 50
    node_count, path_count = np.load("np_counts.npy")

    model = EDKT(config.questions * 2, 
                    config.hidden,
                    config.layers,
                    config.questions,
                    node_count, 
                    path_count, 
                    my_q_len, 
                    config.my_errT_len,
                    config.my_rank_len,
                    max_err_Num,
                    config.err_threshold,
                    device) 

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_func = evaluation.lossFunc(config.questions, config.length, device)
    
    for epoch in range(config.epochs):
        print('【epoch: ' + str(epoch) + '】')
        model, optimize = evaluation.train_epoch(model, train_loader, my_train_qa_loader, my_train_q_loader, my_train_mainerr_loader, my_train_err_loader, my_train_rank_loader, my_train_diff_loader, optimizer, loss_func, config, epoch, device)
        
        
    # torch.save(model, 'model.pth')
    
    
    first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
        model, test_loader, my_test_qa_loader, my_test_q_loader, my_test_mainerr_loader, my_test_err_loader, my_test_rank_loader, my_test_diff_loader, loss_func, device, epoch, config)

    first_total_scores_list.append(first_total_scores)
    scores_list.append(scores)
    first_scores_list.append(first_scores)
    performance_list.append(performance)
    
    print("Average scores of the first attempts:", np.mean(first_total_scores_list,axis=0))
    print("Average scores of all attempts:", np.mean(performance_list,axis=0))
    

if __name__ == '__main__':
    main()
