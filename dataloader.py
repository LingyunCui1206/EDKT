import torch
import torch.utils.data as Data
from readdata import data_reader

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_loader(batch_size, num_of_questions, max_step):
    handle = data_reader(os.path.join(BASE_DIR, "data", "train.csv"),
                         os.path.join(BASE_DIR, "data", "val.csv"),
                         os.path.join(BASE_DIR, "data", "test.csv"), max_step,
                        num_of_questions)

    getTrain = handle.get_train_data()
    getTest = handle.get_test_data()

    dtrain = torch.tensor(getTrain[0].astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(getTest[0].astype(float).tolist(),
                         dtype=torch.float32)
    
    mytrain_qa = torch.tensor(getTrain[1])
    mytest_qa = torch.tensor(getTest[1])
    
    mytrain_q = torch.tensor(getTrain[2])
    mytest_q = torch.tensor(getTest[2])
    
    my_q_len = getTest[3]
    
    mytrain_main_error = torch.tensor(getTrain[4].astype(int).tolist(), dtype=torch.int8)
    mytest_main_error = torch.tensor(getTest[4].astype(int).tolist(), dtype=torch.int8)
    
    mytrain_error = torch.tensor(getTrain[5].astype(int).tolist(),
                          dtype=torch.int8)
    mytest_error = torch.tensor(getTest[5].astype(int).tolist(),
                          dtype=torch.int8)
    
    mytrain_rank = torch.tensor(getTrain[6])
    mytest_rank = torch.tensor(getTest[6])
    
    train_err_Num = getTrain[7]
    test_err_Num = getTest[7]
    
    mytrain_diff = torch.tensor(getTrain[8])
    mytest_diff = torch.tensor(getTest[8])

    train_loader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    
    my_train_qa_loader = Data.DataLoader(mytrain_qa, batch_size=batch_size, shuffle=False)
    my_test_qa_loader = Data.DataLoader(mytest_qa, batch_size=batch_size, shuffle=False)
    
    my_train_q_loader = Data.DataLoader(mytrain_q, batch_size=batch_size, shuffle=False)
    my_test_q_loader = Data.DataLoader(mytest_q, batch_size=batch_size, shuffle=False)
    
    my_train_mainerr_loader = Data.DataLoader(mytrain_main_error, batch_size=batch_size, shuffle=False)
    my_test_mainerr_loader = Data.DataLoader(mytest_main_error, batch_size=batch_size, shuffle=False)
    
    my_train_err_loader = Data.DataLoader(mytrain_error, batch_size=batch_size, shuffle=False)
    my_test_err_loader = Data.DataLoader(mytest_error, batch_size=batch_size, shuffle=False)
    
    my_train_rank_loader = Data.DataLoader(mytrain_rank, batch_size=batch_size, shuffle=False)
    my_test_rank_loader = Data.DataLoader(mytest_rank, batch_size=batch_size, shuffle=False)
    
    my_train_diff_loader = Data.DataLoader(mytrain_diff, batch_size=batch_size, shuffle=False)
    my_test_diff_loader = Data.DataLoader(mytest_diff, batch_size=batch_size, shuffle=False)
    
    
    return train_loader, test_loader, my_train_qa_loader, my_test_qa_loader, my_train_q_loader, my_test_q_loader, my_q_len, my_train_mainerr_loader, my_test_mainerr_loader, my_train_err_loader, my_test_err_loader, my_train_rank_loader, my_test_rank_loader, max(train_err_Num, test_err_Num), my_train_diff_loader, my_test_diff_loader
