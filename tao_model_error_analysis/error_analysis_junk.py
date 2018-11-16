import numpy as np
f_ranks = open('ranks.txt','w')
solved_files_solved_with_support_docs = eval(open("new_relevant_files_solved_with_support_docs").read())
solved_with_support = []
for s in solved_files_solved_with_support_docs:
    s = s[s.rfind('/')+1:]
    solved_with_support.append(s)
f_train = open('train.txt','r')
train_lines = f_train.readlines()
f_train.close()
f_test = open('tao_test_lucene_200_5.txt','r')
test_lines = f_test.readlines()
f_test.close()
TOP_N = 1

question_lists = []
candidates_lists = []
correct_list = []

train_correct_list = []
for line in train_lines:
    line = line.strip()
    line_list = line.split('\t')
    train_correct_list.append(line_list[1])


map_support_doc_cosine = {}

sup_doc_file = open('support_doc_cosine.txt','r')
line_sup_doc = sup_doc_file.readlines()
for line in line_sup_doc:
    line = line.strip()
    line_list = line.split()
    map_support_doc_cosine[tuple([line_list[0],line_list[1]])] = float(line_list[2])
    map_support_doc_cosine[tuple([line_list[1],line_list[0]])] = float(line_list[2])
    map_support_doc_cosine[tuple([line_list[0],line_list[0]])] = 0.0
    map_support_doc_cosine[tuple([line_list[1],line_list[1]])] = 0.0

for line in test_lines:
    line = line.strip().split('\t')
    candidates_lists.append(line[2].split())
    correct_list.append(line[1])
    question_lists.append(line[0])

print "Read answers from input test file"

f_results = open('tao_test_lucene_200_5.txt_evaluate','r')
result_lines = f_results.readlines()
f_results.close()
candidate_ranks = []

num_lines = 0
array_len = 0

for line in result_lines:
    line = line.strip()
    if line == "":
        candidate_ranks = []
        continue
    if line[0] == '[':
        line_list = []
        line = line[1:]
        mini_list = [int(k) for k in line.split()]
        line_list += mini_list

    elif line[-1] == ']':
        num_lines += 1
        array_len += 1
        line = line[:-1]
        mini_list = [int(k) for k in line.split()]
        line_list += mini_list
        if len(line_list) != 200:
            pass
            #continue
            #print line
        assert len(line_list) <= 400
        candidate_ranks.append(line_list)

    else:
        mini_list = [int(k) for k in line.split()]
        line_list += mini_list

print "Read data from the output for the test file"
print "Number of candidate ranks %d" %(len(candidate_ranks))
print "Number of correct list %d" %len(correct_list)
assert len(candidate_ranks) == len(correct_list)
correct_avg = 0.0
num_correct = 0.0
wrong_avg = 0.0
num_wrong = 0.0
correct_train_presence = {}
wrong_train_presence = {}
authentic_wrong = 0
authentic_correct = 0

f_error = open('correct_document.txt','w')

for i,question in enumerate(question_lists):
    print "Considering Question %s \n" %(question)
    correct_document_index = candidates_lists[i].index(correct_list[i])
    correct_document_rank = candidate_ranks[i][correct_document_index]
    print "Correct document %s got a rank %d" %(correct_list[i],correct_document_rank)
    #print "File %s got a rank %d" %(question,correct_document_rank+1)
    f_ranks.write(question+','+str(correct_document_rank+1)+'\n')
    top_5_indices = []
    f_error.write(str(correct_document_rank)+'\n')
    for j in range(TOP_N):
        top_5_indices.append(candidate_ranks[i].index(j))
        print "Rank %d for candidate %s" %(j,candidates_lists[i][candidate_ranks[i].index(j)]),
        if correct_document_rank < TOP_N:
            correct_avg += map_support_doc_cosine[tuple([correct_list[i],candidates_lists[i][j]])]
            #correct_train_presence += train_correct_list.count(correct_list[i])
        else:
            wrong_avg += map_support_doc_cosine[tuple([correct_list[i],candidates_lists[i][j]])]
            #wrong_train_presence += train_correct_list.count(correct_list[i])
    if candidate_ranks[i].index(0) < TOP_N:
        num_correct += 1.0
        if correct_list[i] in correct_train_presence:
            correct_train_presence[correct_list[i]] = train_correct_list.count(correct_list[i])
        else:
            correct_train_presence[correct_list[i]] = train_correct_list.count(correct_list[i])
        #correct_train_presence += train_correct_list.count(correct_list[i])
        if question in solved_with_support:
            authentic_correct += 1
    else:
        num_wrong += 1.0
        if correct_list[i] in wrong_train_presence:
            wrong_train_presence[correct_list[i]] = train_correct_list.count(correct_list[i])
        else:
            wrong_train_presence[correct_list[i]] = train_correct_list.count(correct_list[i])
        #wrong_train_presence += train_correct_list.count(correct_list[i])
        if question in solved_with_support:
            authentic_wrong += 1
    print
    print
    print
print "Correct cosine distance %f" %(correct_avg/num_correct)
print "Wrong cosine distance %f" %(wrong_avg/num_wrong)
print "Number of correct %d" %(num_correct)
print "Number of wrong %d" %(num_wrong)
print "Mean correct %d" %(np.mean(correct_train_presence.values()))
print "Mean Wrong %d" %(np.mean(wrong_train_presence.values()))
print "Authentic Correct %d and Authentic Wrong %d" %(authentic_correct,authentic_wrong)
f_error.close()
f_ranks.close()
