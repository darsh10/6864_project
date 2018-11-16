import sys
import numpy as np
import matplotlib.pyplot as plt
difference_train = {}
difference_count = {}

f_train = open(sys.argv[3],'r')
f_train_lines = f_train.readlines()
f_test = open(sys.argv[1],'r')
f_test_lines = f_test.readlines()
f_result = open(sys.argv[2],'r')

train_questions = []
train_candidates = []
train_results = []
for line in f_train_lines:
    line = line.strip()
    line_list = line.split('\t')
    train_results.append(line_list[1])
    train_candidates.append(line_list[2].split(' '))
    train_questions.append(line_list[0])

test_questions = []
test_candidates = []
test_results = []
for line in f_test_lines:
    line = line.strip()
    line_list = line.split('\t')
    test_results.append(line_list[1])
    test_candidates.append(line_list[2].split(' '))
    test_questions.append(line_list[0])

f_test.close()
result_scores = []
for line in f_result:
    line = line.strip()
    if line == "":
        result_scores = []
        continue
    if line[0] == '[':
        line = line[1:]
        cur_ranks = []
        mini_list = line.split()
        for k in mini_list:
            cur_ranks.append(float(k))
    elif line[-1] == ']':
        line = line[:-1]
        mini_list = line.split()
        for k in mini_list:
            cur_ranks.append(float(k))
        assert len(cur_ranks) == len(test_candidates[len(result_scores)])
        result_scores.append(cur_ranks)
    else:
        mini_list = line.split()
        for k in mini_list:
            cur_ranks.append(float(k))
assert len(result_scores) == len(test_candidates)
print len(result_scores)
print len(test_candidates)
float_mrr = 0.0
p_1 = 0.0
p_5 = 0.0
out_of_p_5_error = 0.0
best_wrong_train_count_under_5 = 0.0
correct_train_count_under_5 = 0.0
best_wrong_train_count_out_of_5 = 0.0
correct_train_count_out_of_5 = 0.0
f=open("error.txt","w")
f_p5=open("out_of_p@5_error.txt","w")
for i in range(len(result_scores)):
    correct_answer = test_results[i]
    correct_answer_candidate_index = test_candidates[i].index(correct_answer)
    correct_answer_model_score = result_scores[i][correct_answer_candidate_index]
    correct_answer_model_rank = 0.0
    for k in range(len(result_scores[i])):
        if k==correct_answer_candidate_index:
            continue
        if result_scores[i][k] > correct_answer_model_score:
            correct_answer_model_rank += 1.0
    float_mrr += 1.0/(1.0+correct_answer_model_rank)
    if correct_answer_model_rank < 1:
        p_1 += 1.0
        if correct_answer_model_rank in difference_train:
            difference_train[correct_answer_model_rank]+=0
            difference_count[correct_answer_model_rank]+=1
        else:
            difference_train[correct_answer_model_rank] = 0
            difference_count[correct_answer_model_rank] = 1
    else:
        f.write("\n"+"https://discussions.apple.com/thread/"+str(test_questions[i]+"\n"))
        f.write("https://support.apple.com/en-us/"+test_candidates[i][correct_answer_candidate_index]+" "+str(result_scores[i][correct_answer_candidate_index])+"\n")
        max_ind = -1
        max_score = -1.0
        for k in range(len(result_scores[i])):
            if result_scores[i][k] > correct_answer_model_score:
                f.write("https://support.apple.com/en-us/"+test_candidates[i][k]+" "+str(result_scores[i][k])+"\n")
                if max_score < result_scores[i][k]:
                    max_score = result_scores[i][k]
                    max_ind = k
        f.write("https://support.apple.com/en-us/"+test_candidates[i][max_ind]+" "+str(result_scores[i][max_ind])+"\n")
        correct_train_count_under_5 += train_results.count(test_candidates[i][correct_answer_candidate_index])
        best_wrong_train_count_under_5 += train_results.count(test_candidates[i][max_ind])
        if correct_answer_model_rank in difference_train:
            difference_train[correct_answer_model_rank] += train_results.count(test_candidates[i][max_ind]) - train_results.count(test_candidates[i][correct_answer_candidate_index])
            difference_count[correct_answer_model_rank] += 1
        else:
            difference_train[correct_answer_model_rank] = train_results.count(test_candidates[i][max_ind]) - train_results.count(test_candidates[i][correct_answer_candidate_index])
            difference_count[correct_answer_model_rank] = 1
    if correct_answer_model_rank < 5:
        p_5 += 1.0
    if correct_answer_model_rank >= 5:
        out_of_p_5_error += 1.0
        f_p5.write("\n"+"https://discussions.apple.com/thread/"+str(test_questions[i]+"\n"))
        f_p5.write("https://support.apple.com/en-us/"+test_candidates[i][correct_answer_candidate_index]+" "+str(result_scores[i][correct_answer_candidate_index])+"\n")
        correct_train_count_out_of_5 += train_results.count(test_candidates[i][correct_answer_candidate_index])
        max_ind = -1
        max_score = -1.0
        for k in range(len(result_scores[i])):
            if result_scores[i][k] > correct_answer_model_score:
                f_p5.write("https://support.apple.com/en-us/"+test_candidates[i][k]+" "+str(result_scores[i][k])+"\n")
                if max_score < result_scores[i][k]:
                    max_score = result_scores[i][k]
                    max_ind = k
        f_p5.write("https://support.apple.com/en-us/"+test_candidates[i][max_ind]+" "+str(result_scores[i][max_ind])+"\n")
        best_wrong_train_count_out_of_5 += train_results.count(test_candidates[i][max_ind])
f_p5.close()
                
float_mrr /= len(result_scores)
p_1 /= len(result_scores)
p_5 /= len(result_scores)
print "MRR achieved %f " %float_mrr
print "P@1 %f, P@5 %f" %(p_1,p_5)
print "Best train count %f, correct train count %f - for all that fall out of top 5" %(best_wrong_train_count_out_of_5/out_of_p_5_error,correct_train_count_out_of_5/out_of_p_5_error)
print "Best train count %f, correct train count %f - for all that fall from 2 to infi" %(best_wrong_train_count_under_5,correct_train_count_under_5)
f.close()
print difference_train
print difference_count
X = []
Y = []
for rank,count in difference_count.keys():
    X.append(rank+1)
    Y.append(difference_train[rank]*1.0/count)
plt.plot(X,Y)
plt.savefig('ranks.png')
