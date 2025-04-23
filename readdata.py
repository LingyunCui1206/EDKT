import numpy as np
import itertools
import pandas as pd

from config import Config

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def diff_data():
    config = Config()
    
    main_df_raw = pd.read_csv(
        os.path.join(BASE_DIR, "../Dataset/CodeWorkout", "MainTable.csv"),
        sep=","
    )
    
    main_df = main_df_raw[main_df_raw["EventType"] == "Run.Program"]
    main_df_Ass1 = main_df[main_df["AssignmentID"] == config.assignment]
    
    problems = pd.unique(main_df_Ass1["ProblemID"])
    problems_d = {k:v for (v,k) in enumerate(problems) }
    
    wrong_percent = (
        main_df_Ass1.assign(is_wrong=lambda x: x['Score'] != 1)
        .groupby('ProblemID')['is_wrong']
        .mean()*10 
    ).astype(int).astype(str)
    
    wrong_percent = wrong_percent.rename(index=problems_d)
    
    return wrong_percent


def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token 
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",~,")
        if components[0] in node_word_index:
            starting_node = node_word_index[components[0]]
        else:
            starting_node = node_word_index['UNK']
            
        if len(components)>=2 and components[1] in path_word_index:
            path = path_word_index[components[1]]
        else:
            path = path_word_index['UNK']
            
        if len(components)>=3 and components[2] in node_word_index:
            ending_node = node_word_index[components[2]]
        else:
            ending_node = node_word_index['UNK']
        
        sample_index.append([starting_node,path,ending_node])
    return sample_index

MAX_CODE_LEN = 128
ERR_TYPE_LEN = 12

class data_reader():
    def __init__(self, train_path, val_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def get_data(self, file_path): 
        data = []
        code_df = pd.read_csv(
            os.path.join(BASE_DIR, "data", "labeled_paths.tsv"), 
            sep="\t"
        )
        
        diff_q = diff_data()
        
        my_q_list = code_df.ProblemID.unique().tolist()
        my_q_len = len(my_q_list)
        
        training_students = np.load(
            os.path.join(BASE_DIR, "data", "training_students.npy"),
            allow_pickle=True
        )
        all_training_code = code_df[code_df['SubjectID'].isin(training_students)]['RawASTPath'] 
        
        separated_code = []
        for code in all_training_code:
            if type(code) == str:
                separated_code.append(code.split("@~@"))
        
        node_hist = {}
        path_hist = {}
        
        starting_nodes = []
        path = []
        ending_nodes = []
        
        for paths in separated_code:
            
            starting_nodes = [p.split(",~,")[0] for p in paths]
            path = [p.split(",~,")[1] for p in paths]
            ending_nodes = [p.split(",~,")[2] for p in paths]
            
            nodes = starting_nodes + ending_nodes
            
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
                    
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1

        node_count = len(node_hist)
        path_count = len(path_hist)

        np.save("np_counts.npy", [node_count, path_count])

        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)
        
        question_answer = []
        questions = []
        error = []
        main_error = []
        ranks = []
        diff_question = []
        
        
        with open(file_path, 'r') as file:
            for lent, css, ques, ans, rank, errType in itertools.zip_longest(*[file] * 6):
                
                # print(lent, css, ques, ans, rank, errType)
                
                if ques == None or rank == None or errType == None:
                    continue
                
                lent = int(lent.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                rank = [int(r) for r in rank.strip().strip(',').split(',')]
                
                css = [cs for cs in css.strip().strip(',').split(',')]
                errType = [et for et in errType.strip().strip(',').split(',')]
                
                
                diff_ques = [int(diff_q[q]) for q in ques]
                
                temp = np.zeros(shape=[self.maxstep, 2 * self.numofques+MAX_CODE_LEN*3])
                if lent >= self.maxstep:
                    steps = self.maxstep
                    extra = 0
                    ques = ques[-steps:]
                    ans = ans[-steps:]
                    rank = rank[-steps:]
                    diff_ques = diff_ques[-steps:]
                    css = css[-steps:]
                    errType = errType[-steps:]
                else:
                    steps = lent
                    extra = self.maxstep-steps
                    error_pad = ['12']*extra
                    errType=errType+error_pad
                
                
                
                errTemp = np.zeros(shape=[self.maxstep, ERR_TYPE_LEN],dtype=int)
                mainErr = []
                
                for i in range(self.maxstep):
                    err = errType[i]
                    err_list_str = err.split("@")
                    err_list_str = list(filter(None, err_list_str))
                    err_list = list(map(int, err_list_str))
                    mainErr.append(err_list[0])
                    for j in err_list:
                        if j != 12:
                            errTemp[i][j] += 1 
                
                # print(mainErr)
                # exit()
                    
                
                ques_ans = []

                for j in range(steps):
                    if ans[j] == 1:
                        ques_number = ques[j]
                        temp[j+extra][ques[j]] = 1 
                    else:
                        ques_number = ques[j] + (max(ques) + 1)
                        temp[j+extra][ques[j] + self.numofques] = 1
                    
                    code = code_df[code_df['CodeStateID']==int(css[j])]['RawASTPath'].iloc[0]
                    
                    
                    if type(code) == str:
                        code_paths = code.split("@~@")
                        raw_features = convert_to_idx(code_paths, node_word_index, path_word_index)
                        
                        
                        if len(raw_features) < MAX_CODE_LEN:
                            raw_features += [[0,0,0]]*(MAX_CODE_LEN - len(raw_features))
                        else:
                            raw_features = raw_features[:MAX_CODE_LEN]
                        
                        features = np.array(raw_features).reshape(-1, MAX_CODE_LEN*3)
                        
                        temp[j+extra][2*self.numofques:] = features
                        
                    ques_ans.append(ques_number)

                
                data.append(temp.tolist())
                question_answer.append(ques_ans[:])
                questions.append(ques[:])
                error.append(errTemp.tolist())
                main_error.append(mainErr[:])
                ranks.append(rank[:])
                diff_question.append(diff_ques[:])
                
            
            # qa的padding
            pad_list_qa = []
            pad_list_q = []
            pad_list_rank = []
            pad_list_diff = []
            for i in range(len(question_answer)):
                pad = [20] * (self.maxstep - len(question_answer[i]))
                pad.extend(question_answer[i])
                pad_list_qa.append(pad[:])
                
                pad = [10] * (self.maxstep - len(questions[i]))
                pad.extend(questions[i])
                pad_list_q.append(pad[:])
                
                pad = [12] * (self.maxstep - len(ranks[i]))
                pad.extend(ranks[i])
                pad_list_rank.append(pad[:])
                
                pad = [101] * (self.maxstep - len(diff_question[i]))
                pad.extend(diff_question[i])
                pad_list_diff.append(pad[:])
                
            print('done: ' + str(np.array(data).shape))
            err_num = np.max(main_error)
            
        
        return data, pad_list_qa, pad_list_q, my_q_len, main_error, error, pad_list_rank, err_num, pad_list_diff
    
    def get_train_data(self):
        print('loading train data...')
        train_data, train_ques_ans, train_ques, my_q_len, train_main_error, train_error, train_rank, train_err_num, train_diff = self.get_data(self.train_path)
        val_data, val_ques_ans, val_ques, my_q_len, val_main_error, val_error, val_rank, val_err_num, val_diff = self.get_data(self.val_path)
        return (np.array(train_data+val_data), 
                np.array(train_ques_ans+val_ques_ans),
                np.array(train_ques+val_ques), 
                my_q_len, 
                np.array(train_main_error+val_main_error), 
                np.array(train_error+val_error), 
                np.array(train_rank+val_rank), 
                max(train_err_num, val_err_num), 
                np.array(train_diff+val_diff))
    
    
    def get_test_data(self):
        print('loading test data...')
        test_data, test_ques_ans, test_ques, my_q_len, test_main_error, test_error, test_rank, test_err_num, test_diff = self.get_data(self.test_path)
        return (np.array(test_data), 
                np.array(test_ques_ans), 
                np.array(test_ques), 
                my_q_len, 
                np.array(test_main_error), 
                np.array(test_error), 
                np.array(test_rank),
                test_err_num,
                np.array(test_diff))
