from tqdm.contrib import tzip
import torch
import numpy as np

import torch.nn as nn
from sklearn import metrics

def performance_granular(batch, pred, ground_truth, prediction, epoch, config):
    
    
    preds = {k:[] for k in range(config.questions)}
    gts = {k:[] for k in range(config.questions)}
    first_preds = {k:[] for k in range(config.questions)}
    first_gts = {k:[] for k in range(config.questions)}
    scores = {}
    first_scores = {}

    
    for s in range(pred.shape[0]):
        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions*2])
        temp = pred[s][:config.length-1].mm(delta.T)
        index = torch.tensor([[i for i in range(config.length-1)]],
                             dtype=torch.long)
        p = temp.gather(0, index)[0].detach().cpu().numpy()
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]).sum(1) + 1) // 2)[1:].detach().cpu().numpy()

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta.detach().cpu().numpy()[i:]
                break
        
        
        for i in range(len(p)):
            for j in range(config.questions):
                if delta[i,j] == 1:
                    preds[j].append(p[i])
                    gts[j].append(a[i])
                    if i == 0 or delta[i-1,j] != 1:
                        first_preds[j].append(p[i])
                        first_gts[j].append(a[i])
                        
    first_total_gts = []
    first_total_preds = []
    print("=======================================================")
    for j in range(config.questions):
        f1 = metrics.f1_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        recall = metrics.recall_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        precision = metrics.precision_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        acc = metrics.accuracy_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        try:
            auc = metrics.roc_auc_score(gts[j], preds[j])
        except ValueError:
            auc = 0.5
        scores[j]=[auc,f1,recall,precision,acc]
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print('【problem '+str(j)+'】 auc: ' + str(auc) + '| f1: ' + str(f1) + '| recall: ' + str(recall) +
          '| precision: ' + str(precision) + '| acc: ' +
                str(acc))
        
        
        first_f1 = metrics.f1_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_recall = metrics.recall_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_precision = metrics.precision_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_acc = metrics.accuracy_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        try:
            first_auc = metrics.roc_auc_score(first_gts[j], first_preds[j])
        except ValueError:
            first_auc = 0.5
            
        first_total_gts.extend(first_gts[j])
        first_total_preds.extend(first_preds[j])
        
        first_scores[j]=[first_auc,first_f1,first_recall,first_precision,first_acc]
        print("-------------------------------------------------------")
        print('【First prediction for problem '+str(j)+'】 auc: ' + str(first_auc) + '| f1: ' + str(first_f1) + '| recall: ' + str(first_recall) + '| precision: ' + str(first_precision) + '| acc: ' + str(first_acc))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    
    print("=======================================================")
    
    print('|auc: ' + str(auc) + '| f1: ' + str(f1) + '| recall: ' + str(recall) +
          '| precision: ' + str(precision) + '| acc: ' +
                str(acc))
    print("=======================================================")
    
    
    
    
    first_total_f1 = metrics.f1_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_recall = metrics.recall_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_precision = metrics.precision_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_acc = metrics.accuracy_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    try:
        first_total_auc = metrics.roc_auc_score(first_total_gts, first_total_preds)
    except ValueError:
        first_total_auc = 0.5
    
    first_total_scores = [first_total_auc,first_total_f1,first_total_recall,first_total_precision,first_total_acc]
    
    return first_total_scores, first_scores, scores, [auc,f1,recall,precision,acc]
        

class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device


    def forward(self, pred, batch):
        
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')

        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:self.num_of_questions] + batch[student][:, self.num_of_questions:self.num_of_questions*2]
            temp = pred[student][:self.max_step-1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step-1)]],
                                 dtype=torch.long)
            p = temp.gather(0, index)[0]
            
            a = (((batch[student][:, 0:self.num_of_questions] - batch[student][:, self.num_of_questions:self.num_of_questions*2]).sum(1) + 1) // 2)[1:]
            
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break

            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth
    
    

def train_epoch(model, trainLoader, my_train_qa_loader, my_train_q_loader, my_train_mainerr_loader, my_train_err_loader, my_train_rank_loader, my_train_diff_loader, optimizer, loss_func, config, epoch, device):
    model.to(device)
    model.train()
    
    # train_loss = 0
    
    for batch, my_batch_qa, my_batch_q, my_batch_main_err, my_batch_err, my_batch_rank, my_batch_diff in tzip(trainLoader, my_train_qa_loader, my_train_q_loader, my_train_mainerr_loader, my_train_err_loader, my_train_rank_loader, my_train_diff_loader, desc='Training:    ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)
        my_batch_qa_new = my_batch_qa[:,:-1].to(device) 
        my_batch_q_new = my_batch_q[:,:-1].to(device)
        my_batch_err_new = my_batch_err[:,:-1].to(device) 
        my_batch_rank_new = my_batch_rank[:,:-1].to(device) 
        my_batch_diff_new = my_batch_diff[:,:-1].to(device) 
        
        pred, h, h_EC, h_PD, qa_emb, code_vectors, rnn_input, problem_diff, personal_diff, complie_err_gain, run_err_gain = model(batch_new, my_batch_qa_new, my_batch_q_new, my_batch_err_new, my_batch_rank_new, my_batch_diff_new)
        loss, p, a = loss_func(pred, batch[:,:,:config.questions*2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    #     train_loss += loss.item()
    # train_loss /= len(trainLoader)
    
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         ground_truth = torch.tensor([])
#         prediction = torch.tensor([])
#         full_data = torch.tensor([])
#         preds = torch.tensor([])
#         batch_n = 0
        
#         for batch, my_batch_qa, my_batch_q, my_batch_main_err, my_batch_err, my_train_rank in tzip(val_loader, my_val_qa_loader, my_val_q_loader, my_val_mainerr_loader, my_val_err_loader, my_val_rank_loader, desc='Vallling:    ', mininterval=2):
#             batch_new = batch[:,:-1,:].to(device) 
#             my_batch_qa_new = my_batch_qa[:,:-1].to(device) 
#             my_batch_q_new = my_batch_q[:,:-1].to(device)
#             my_batch_err_new = my_batch_err[:,:-1].to(device) 
#             my_train_rank_new = my_train_rank[:,:-1].to(device) 

#             pred = model(batch_new, my_batch_qa_new, my_batch_q_new, my_batch_err_new, my_train_rank_new)
#             loss, p, a = loss_func(pred, batch[:,:,:config.questions*2])
#             val_loss += loss.item()
            
#             prediction = torch.cat([prediction, p])
#             ground_truth = torch.cat([ground_truth, a])
            
#     val_loss /= len(val_loader)
#     v_auc = val_auc(ground_truth, prediction, config)
    
#     print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {v_auc:.4f}")
    
    
    return model, optimizer


def test_epoch(model, testLoader, my_test_qa_loader, my_test_q_loader, my_test_mainerr_loader, my_test_err_loader, my_test_rank_loader, my_test_diff_loader, loss_func, device, epoch, config):
    
    # model = torch.load('model.pth')
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        ground_truth = torch.tensor([])
        prediction = torch.tensor([])
        full_data = torch.tensor([])
        preds = torch.tensor([])
        batch_n = 0
        for batch, my_batch_qa, my_batch_q, my_batch_main_err, my_batch_err, my_batch_rank, my_batch_diff in tzip(testLoader, my_test_qa_loader, my_test_q_loader, my_test_mainerr_loader, my_test_err_loader, my_test_rank_loader, my_test_diff_loader, desc='Training:    ', mininterval=2):
            batch_new = batch[:,:-1,:].to(device)
            my_batch_qa_new = my_batch_qa[:,:-1].to(device)
            my_batch_q_new = my_batch_q[:,:-1].to(device)
            my_batch_err_new = my_batch_err[:,:-1].to(device)
            my_batch_rank_new = my_batch_rank[:,:-1].to(device) 
            my_batch_diff_new = my_batch_diff[:,:-1].to(device) 
            
            pred, h, h_EC, h_PD, qa_emb, code_vectors, rnn_input, problem_diff, personal_diff, complie_err_gain, run_err_gain  = model(batch_new, my_batch_qa_new, my_batch_q_new, my_batch_err_new, my_batch_rank_new, my_batch_diff_new)
            loss, p, a = loss_func(pred, batch)
            
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
            full_data = torch.cat([full_data, batch])
            preds = torch.cat([preds, pred.cpu()])
            batch_n += 1

    return performance_granular(full_data, preds, ground_truth, prediction, epoch, config)


