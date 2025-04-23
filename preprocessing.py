import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from errorType import getErrorType
import re


errorTpyes = getErrorType()

main_df_raw = pd.read_csv("../Dataset/CodeWorkout/MainTable.csv")

studen = pd.unique(main_df_raw["SubjectID"])

Assi = pd.unique(main_df_raw["AssignmentID"])

main_df_errLine = main_df_raw[main_df_raw["EventType"] == "Compile.Error"]

main_df = main_df_raw[main_df_raw["EventType"] == "Run.Program"]
main_df = main_df[main_df["AssignmentID"] == 439]

students = pd.unique(main_df["SubjectID"])

problems = pd.unique(main_df["ProblemID"])
problems_d = {k:v for (v,k) in enumerate(problems) }

# print(problems_d)

def getProblemMap():
    return problems_d

def classify_score(score):
    if score == 0:
        return '0'
    elif score == 1:
        return '1'
    elif np.isnan(score):
        return '12'  #PAD
    else:
        scoreRank = int(score * 10) + 2
        errorRank = 13 - scoreRank
        return str(errorRank)
    

count_df = main_df.groupby(['SubjectID', 'ProblemID']).size().reset_index(name='Attempts')
print(f"每个学生做每道题的尝试次数为: ")
print(f"{count_df}")

main_df['ServerTimestamp'] = pd.to_datetime(main_df['ServerTimestamp'])
time_spent = (
    main_df
    .sort_values(['SubjectID', 'ProblemID', 'ServerTimestamp'])
    .groupby(['SubjectID', 'ProblemID'])['ServerTimestamp']
    .agg(
        first_time='min',
        last_time='max'
    )
    .reset_index()
)
time_spent['total_time'] = time_spent['last_time'] - time_spent['first_time']
time_spent['total_seconds'] = time_spent['total_time'].dt.total_seconds().round()
time_df = time_spent[['SubjectID', 'ProblemID', 'total_time', 'total_seconds']]
# print(time_df)

d = {}
for s in students:
    d[s] = {} 
    df = main_df[main_df["SubjectID"] == s]
    d[s]["length"] = len(df)
    d[s]["Problems"] = [str(problems_d[i]) for i in df["ProblemID"]]
    
    d[s]["Result"] = list((df["Score"]==1).astype(int).astype(str)) 
    
    d[s]["Rank"] = list(df['Score'].apply(classify_score))
    
    d[s]["CodeStates"] = list(df["CodeStateID"])
    
    print(count_df[(count_df["SubjectID"] == s) & (count_df["ProblemID"] == q)]["Attempts"])
    print(count_df[(count_df["SubjectID"] == s) & (count_df["ProblemID"] == q)]["Attempts"].values)
    
    d[s]["Attempts"] = [str(count_df[(count_df["SubjectID"] == s) & (count_df["ProblemID"] == q)]["Attempts"].values) for q in d[s]["Problems"]]
    
    d[s]["Time"] = [str(time_df[(time_df["SubjectID"] == s) & (time_df["ProblemID"] == q)]["total_seconds"].values) for q in d[s]["Problems"]]
    
    
    pattern = r"line (\d+): (.+)"
    et = []
    
    err_df = main_df_errLine[main_df_errLine["SubjectID"] == s]
    for index, codeID in enumerate(d[s]["CodeStates"]):
        et_per = ''
        if d[s]["Result"][index] == '1':
            et.append('@0')
            continue
        errorLine = err_df[err_df["CodeStateID"] == codeID]
        if len(errorLine) != 0:
            CompileMessage = errorLine["CompileMessageData"]
            for message in CompileMessage:
                if not isinstance(message, str) or message == 'nan':
                    et_per=et_per+'@'+'0'
                else:
                    match = re.match(pattern, message)
                    if match:
                        line_number = match.group(1)
                        error_description = match.group(2)
                        for index, error in enumerate(errorTpyes):
                            if error_description in error:
                                et_per=et_per+'@'+str(index)
                # break
            et.append(et_per)
        else:
            et.append('@12')
        
    d[s]['ErrorTypes'] = et
            
    
    
train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)



np.save("./data/training_students.npy", train_val_s)
np.save("./data/testing_students.npy", test_s)

if not os.path.isdir("./data"):
    os.mkdir("./data")

file_test = open("./data/test.csv","w")
for s in test_s:
    if d[s]['length']>0:
        file_test.write(str(d[s]['length']))
        file_test.write(",\n")
        file_test.write(",".join(str(code_state) for code_state in d[s]['CodeStates']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Problems']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Result']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Rank']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['ErrorTypes']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Attempts']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Time']))
        file_test.write(",\n")

        
# for fold in range(100):
train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=1)

file_train = open("./data/train.csv","w")
for s in train_s:
    if d[s]['length']>0:
        file_train.write(str(d[s]['length']))
        file_train.write(",\n")
        file_train.write(",".join(str(code_state) for code_state in d[s]['CodeStates']))
        file_train.write(",\n")
        file_train.write(",".join(d[s]['Problems']))
        file_train.write(",\n")
        file_train.write(",".join(d[s]['Result']))
        file_train.write(",\n")
        file_train.write(",".join(d[s]['Rank']))
        file_train.write(",\n")
        file_train.write(",".join(d[s]['ErrorTypes']))
        file_train.write(",\n")
        file_test.write(",".join(d[s]['Attempts']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Time']))
        file_test.write(",\n")
        


file_val = open("./data/val.csv","w")
for s in val_s:
    if d[s]['length']>0:
        file_val.write(str(d[s]['length']))
        file_val.write(",\n")
        file_val.write(",".join(str(code_state) for code_state in d[s]['CodeStates']))
        file_val.write(",\n")
        file_val.write(",".join(d[s]['Problems']))
        file_val.write(",\n")
        file_val.write(",".join(d[s]['Result']))
        file_val.write(",\n")
        file_val.write(",".join(d[s]['Rank']))
        file_val.write(",\n")
        file_val.write(",".join(d[s]['ErrorTypes']))
        file_val.write(",\n")
        file_test.write(",".join(d[s]['Attempts']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Time']))
        file_test.write(",\n")