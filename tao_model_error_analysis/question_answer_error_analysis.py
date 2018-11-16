from graph import Graph
import sys
sys.path.append('..')
import numpy as np
#import matplotlib.pyplot as plt
from baseline.baseline_new import Baseline
difference_train = {}
difference_count = {}
num_candidates = 100000000
import pickle
#sys.setrecursionlimit(1500)
try:
    f_train = open(sys.argv[3],'r')
    f_train_lines = f_train.readlines()
    f_test = open(sys.argv[1],'r')
    f_test_lines = f_test.readlines()
    f_result = open(sys.argv[2],'r')
    if len(sys.argv) > 4:
        num_candidates = int(sys.argv[4])

except IndexError:
    print "run with command: python question_answer_error_analysis.py test.txt test.txt_evaluate train.txt"
    sys.exit()

bsl = Baseline(sys.maxint)
bsl.populate_data()
DATA_PATH="../DATA_short/"

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
    if " " in line_list[1]:
        line_list[1] = line_list[1].split()
    else:
        line_list[1] = [line_list[1]]
    print line_list[0],line_list[1][-1]
    assert line_list[0]+"_ans" == line_list[1][-1]
    test_results.append(line_list[1])
    current_test_candidates = line_list[2].split(' ')
    for k in line_list[1]:
        if k not in current_test_candidates:
            current_test_candidates.append(k)
    test_candidates.append(current_test_candidates)
    test_questions.append(line_list[0])

f_test.close()

test_questions = test_questions[:num_candidates]
test_candidates = test_candidates[:num_candidates]
test_results = test_results[:num_candidates]
result_scores = []
#print "Number of lines %d" %len(f_result.readlines())
for line in f_result.readlines()[:num_candidates+1]:
    line = line.strip()
    if line == "":
        result_scores = []
        continue
    if line[0] == '[':
        line = line[1:]
        cur_ranks = []
    #    mini_list = line.split()
    #    for k in mini_list:
    #        if k[-1] == ',':
    #            k = k[:-1]
    #        cur_ranks.append(float(k))
    #elif line[-1] == ']':
        line = line[:-1]
        mini_list = line.split()
        for k in mini_list:
            if k[-1] == ',':
                k = k[:-1]
            cur_ranks.append(float(k))
        print len(cur_ranks),len(test_candidates[len(result_scores)])
        assert len(cur_ranks) == len(test_candidates[len(result_scores)])
        result_scores.append(cur_ranks)
        #print len(result_scores)
    #else:
    #    mini_list = line.split()
    #    for k in mini_list:
    #        cur_ranks.append(float(k))
print len(result_scores), len(test_candidates)
assert len(result_scores) == len(test_candidates)
#print len(result_scores)
#print len(test_candidates)
float_mrr = 0.0
p_1 = 0.0
p_5 = 0.0
out_of_p_5_error = 0.0
best_wrong_train_count_under_5 = 0.0
correct_train_count_under_5 = 0.0
best_wrong_train_count_out_of_5 = 0.0
correct_train_count_out_of_5 = 0.0
file_error = "error.txt"
f=open(file_error,"w")
file_out_of_5 = "out_of_p@5_error.txt"
f_p5=open(file_out_of_5,"w")
answer_top_counts = {}
interesting_answer_counts = {}
answers_considered = 0
question_wrong = pickle.load(open("question_wrong.p","r"))

for i in range(len(result_scores)):
    if (i+1)%1000 == 0:
        print "Error analysis extraction index %d" %(i+1)
    best_correct_answer = ""
    best_correct_score = -1.0
    best_correct_index = -1
    for j in range(len(test_results[i])):
        correct_answer = test_results[i][j]
        correct_answer_candidate_index = test_candidates[i].index(correct_answer)
        correct_answer_model_score = result_scores[i][correct_answer_candidate_index]
        if best_correct_score < correct_answer_model_score:
            best_correct_score = correct_answer_model_score
            best_correct_answer = correct_answer
            best_correct_index = correct_answer_candidate_index

    correct_answer = best_correct_answer
    correct_answer_model_score = best_correct_score
    correct_answer_candidate_index = best_correct_index
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
        if correct_answer not in answer_top_counts:
            answers_considered += 1
            answer_top_counts[correct_answer] = 1
        else:
            answers_considered += 1
            answer_top_counts[correct_answer] += 1
        if correct_answer not in interesting_answer_counts:
            interesting_answer_counts[correct_answer] = 1
        else:
            interesting_answer_counts[correct_answer] += 1
    else:
        max_ind = -1
        max_score = -1.0
        answer_higherscore = {}
        for k in range(len(result_scores[i])):
            if result_scores[i][k] > correct_answer_model_score:
                answer_higherscore[test_candidates[i][k][:test_candidates[i][k].find('_')]] = result_scores[i][k]                      #f.write("https://discussions.apple.com/thread/"+test_candidates[i][k][:test_candidates[i][k].find('_')]+" "+str(result_scores[i][k])+"\n")
                if max_score < result_scores[i][k]:
                    max_score = result_scores[i][k]
                    max_ind = k
        wrong_answer = False
        if test_questions[i] in question_wrong and test_candidates[i][max_ind][:test_candidates[i][max_ind].find('_')] in question_wrong[test_questions[i]]:
            wrong_answer = True
        if not wrong_answer:
            f.write("Question:\n"+"https://discussions.apple.com/thread/"+str(test_questions[i]+"\n"))
            f.write(bsl.solved_files_question_text[bsl.solved_files.index(DATA_PATH+test_questions[i])]+"\n")
            f.write("https://discussions.apple.com/thread/"+test_candidates[i][correct_answer_candidate_index][:test_candidates[i][correct_answer_candidate_index].find('_')]+" "+str(result_scores[i][correct_answer_candidate_index])+"\n")
            f.write(bsl.solved_files_question_answer[bsl.solved_files.index(DATA_PATH+test_questions[i])]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+test_questions[i]]]+"\n")
        #max_ind = -1
        #max_score = -1.0
        #answer_higherscore = {}
        #for k in range(len(result_scores[i])):
        #    if result_scores[i][k] > correct_answer_model_score:
        #        answer_higherscore[test_candidates[i][k][:test_candidates[i][k].find('_')]] = result_scores[i][k]
                #f.write("https://discussions.apple.com/thread/"+test_candidates[i][k][:test_candidates[i][k].find('_')]+" "+str(result_scores[i][k])+"\n")
        #        if max_score < result_scores[i][k]:
        #            max_score = result_scores[i][k]
        #            max_ind = k
        if not wrong_answer:
            f.write("https://discussions.apple.com/thread/"+test_candidates[i][max_ind][:test_candidates[i][max_ind].find('_')]+" "+str(result_scores[i][max_ind])+"\n")
        if test_candidates[i][max_ind] not in answer_top_counts:
            answers_considered += 1
            answer_top_counts[test_candidates[i][max_ind]] = 1
        else:
            answers_considered += 1
            answer_top_counts[test_candidates[i][max_ind]] += 1
        
        if not wrong_answer:
            f.write(bsl.solved_files_question_answer[bsl.solved_files.index(DATA_PATH+test_candidates[i][max_ind][:test_candidates[i][max_ind].find('_')])]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+test_candidates[i][max_ind][:test_candidates[i][max_ind].find('_')]]]+"\n")
        for answer in sorted(answer_higherscore, key=answer_higherscore.get, reverse = False):
            if not wrong_answer:
                f.write("https://discussions.apple.com/thread/"+answer+" "+str(answer_higherscore[answer])+"\n")
                f.write(bsl.solved_files_question_answer[bsl.solved_files.index(DATA_PATH+answer)]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+answer]]+"\n")
            if answer+"_ans" not in interesting_answer_counts:
                interesting_answer_counts[answer+"_ans"] = 1
            else:
                interesting_answer_counts[answer+"_ans"] += 1
        correct_train_count_under_5 += train_results.count(test_candidates[i][correct_answer_candidate_index])
        best_wrong_train_count_under_5 += train_results.count(test_candidates[i][max_ind])
        if correct_answer_model_rank in difference_train:
            difference_train[correct_answer_model_rank] += train_results.count(test_candidates[i][max_ind]) - train_results.count(test_candidates[i][correct_answer_candidate_index])
            difference_count[correct_answer_model_rank] += 1
        else:
            difference_train[correct_answer_model_rank] = train_results.count(test_candidates[i][max_ind]) - train_results.count(test_candidates[i][correct_answer_candidate_index])
            difference_count[correct_answer_model_rank] = 1
    if correct_answer_model_rank < 10:
        p_5 += 1.0
    if correct_answer_model_rank >= 5:
        out_of_p_5_error += 1.0
        f_p5.write("\n"+"https://discussions.apple.com/thread/"+str(test_questions[i]+"\n"))
        f_p5.write("https://discussions.apple.com/thread/"+test_candidates[i][correct_answer_candidate_index][:test_candidates[i][correct_answer_candidate_index].find('_')]+" "+str(result_scores[i][correct_answer_candidate_index])+"\n")
        f_p5.write("https://discussions.apple.com/thread/"+test_candidates[i][correct_answer_candidate_index][:test_candidates[i][correct_answer_candidate_index].find('_')]+" "+str(result_scores[i][correct_answer_candidate_index])+"\n")
        f_p5.write(bsl.solved_files_question_answer[bsl.solved_files.index(DATA_PATH+test_questions[i])]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+test_questions[i]]]+"\n")
        correct_train_count_out_of_5 += train_results.count(test_candidates[i][correct_answer_candidate_index])
        max_ind = -1
        max_score = -1.0
        answer_higherscore = {}
        for k in range(len(result_scores[i])):
            if result_scores[i][k] > correct_answer_model_score:
                f_p5.write("https://discussions.apple.com/thread/"+test_candidates[i][k][:test_candidates[i][k].find('_')]+" "+str(result_scores[i][k])+"\n")
                answer_higherscore[test_candidates[i][k][:test_candidates[i][k].find('_')]] = result_scores[i][k]
                if max_score < result_scores[i][k]:
                    max_score = result_scores[i][k]
                    max_ind = k
        f_p5.write("https://discussions.apple.com/thread/"+test_candidates[i][max_ind][:test_candidates[i][max_ind].find('_')]+" "+str(result_scores[i][max_ind])+"\n")
        for answer in sorted(answer_higherscore, key = answer_higherscore.get, reverse = False):
            f_p5.write("https://discussions.apple.com/thread/"+answer+" "+str(answer_higherscore[answer])+"\n")
            f_p5.write(bsl.solved_files_question_answer[bsl.solved_files.index(DATA_PATH+answer)]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+answer]]+"\n")
        best_wrong_train_count_out_of_5 += train_results.count(test_candidates[i][max_ind])
f_p5.close()
                
float_mrr /= len(result_scores)
p_1 /= len(result_scores)
p_5 /= len(result_scores)
print "MRR achieved %f " %float_mrr
print "P@1 %f, P@5 %f" %(p_1,p_5)
print "Best train count %f, correct train count %f - for all that fall out of top 5" %(best_wrong_train_count_out_of_5/out_of_p_5_error,correct_train_count_out_of_5/out_of_p_5_error)
print "Best train count %f, correct train count %f - for all that fall from 2 to infi" %(best_wrong_train_count_under_5,correct_train_count_under_5)
print "Files written out are %s and %s" %(file_error,file_out_of_5)
f.close()
print "Total number of top occurying answers are %d of %d questions" %(len(answer_top_counts),answers_considered)
print "Total number of interesting answers are %d of %d questions" %(len(interesting_answer_counts),answers_considered)
sys.exit()
top_counts_occurences = {}
interesting_counts_occurences = {}
for answer in sorted(answer_top_counts, key=answer_top_counts.get, reverse=True):
    #print answer,answer_top_counts[answer]
    if answer_top_counts[answer] not in top_counts_occurences:
        top_counts_occurences[answer_top_counts[answer]] = 1
    else:
        top_counts_occurences[answer_top_counts[answer]] += 1
#print top_counts_occurences
for answer in sorted(interesting_answer_counts, key=interesting_answer_counts.get, reverse=True):
    if answer not in answer_top_counts:
        continue
    if interesting_answer_counts[answer] not in interesting_counts_occurences:
        interesting_counts_occurences[interesting_answer_counts[answer]] =1
    else:
        interesting_counts_occurences[interesting_answer_counts[answer]] +=1
#print interesting_counts_occurences
#nodes = answer_top_counts.keys()
nodes = interesting_answer_counts.keys()
edges = {}
edge_counts = {}
reverse_edges = {}
number_of_meetings = 0
for i in range(len(result_scores)):
    if (i+1)%1000==0:
        print "Generating graph at index %d" %(i+1)
    all_better_answers = []
    all_better_answer_scores = []
    candidates = test_candidates[i]
    correct_answer = test_results[i]
    candidate_scores = result_scores[i]
    candidate_scores_map = {}
    correct_score = result_scores[i][candidates.index(correct_answer)]
    for ind,candidate in enumerate(candidates):
        candidate_scores_map[candidate] = candidate_scores[ind]
    ordered_answers = []
    for candidate in sorted(candidate_scores_map, key=candidate_scores_map.get, reverse=True):
        if candidate == correct_answer:
            break
        ordered_answers.append(candidate)
    for i,answer_i in enumerate(ordered_answers):
        #if answer_i not in answer_top_counts:
        if answer_i not in interesting_answer_counts:
            continue
        for j,answer_j in enumerate(ordered_answers):
            if j<=i:
                continue
            #if answer_j not in answer_top_counts:
            if answer_j not in interesting_answer_counts:
                continue
            #print "Possibility of a union"
            number_of_meetings += 1
            if tuple(sorted([answer_i,answer_j])) not in edges:
                edges[tuple(sorted([answer_i,answer_j]))] = 1.0/(0.0000000001+abs(candidate_scores_map[answer_i] - candidate_scores_map[answer_j]))
                edge_counts[tuple(sorted([answer_i,answer_j]))] = 1
                reverse_edges[tuple(sorted([answer_j,answer_i]))] = candidate_scores_map[answer_i] - candidate_scores_map[answer_j]
            else:
                edges[tuple(sorted([answer_i,answer_j]))] += 1.0/(1.0+abs(candidate_scores_map[answer_i] - candidate_scores_map[answer_j]))
                edge_counts[tuple(sorted([answer_i,answer_j]))] += 1
                reverse_edges[tuple(sorted([answer_j,answer_i]))] += candidate_scores_map[answer_i] - candidate_scores_map[answer_j]

print "Total number of edges is %d" %len(edges)
print "Total number of meetings is %d" %number_of_meetings
#Removing trivial nodes
ignore_nodes = set()
for i,answer_i in enumerate(nodes):
    for j,answer_j in enumerate(nodes):
        if j<=i:
            continue
        if answer_j in ignore_nodes:
            continue
        if tuple([answer_i,answer_j]) in edges and tuple([answer_j,answer_i]) not in edges:
            ignore_nodes.add(answer_j)
        if  tuple([answer_i,answer_j]) not in edges and tuple([answer_j,answer_i]) in edges:
            ignore_nodes.add(answer_i)

print "Number of nodes is %d" %len(nodes)
print "Number of nodes removed on account of being trivial is %d" %len(list(ignore_nodes))
selected_nodes = set()
ignore_nodes = set()
#prune the number of edges
to_be_deleted = []
for node in nodes:
    #if answer_top_counts[node] < NODE_THRESHOLD:
    if interesting_answer_counts[node] < NODE_THRESHOLD:
        to_be_deleted.append(node)

for node in to_be_deleted:
    nodes.remove(node)
to_be_deleted = []

node_set = set(nodes)
for edge in edges:
    #if edge_counts[edge] < EDGE_THRESHOLD:
    if (edge[0] not in node_set or edge[1] not in node_set) or edge_counts[edge]<EDGE_THRESHOLD:
        to_be_deleted.append(edge)

original_num_edges = len(edges)
for edge in to_be_deleted:
    del edges[edge]

new_num_edges = len(edges)

print "%d new edges from %d old edges" %(new_num_edges,original_num_edges)
node_parents = {}
while len(edges)>0 and len(nodes)>(len(selected_nodes.union(ignore_nodes))):
    node_edge_score = {}
    max_score = 0.0
    max_node = -1
    for edge in edges:
        node_1,node_2 = edge[0],edge[1]
        assert node_1 not in selected_nodes
        assert node_2 not in selected_nodes
        if node_1 in node_parents and node_2 in node_parents and len(node_parents[node_1].union(node_parents[node_2])) > 0:
            continue
        if node_1 not in node_edge_score:
            node_edge_score[node_1] = edges[edge]
        else:
            node_edge_score[node_1] += edges[edge]
        if node_2 not in node_edge_score:
            node_edge_score[node_2] = edges[edge]
        else:
            node_edge_score[node_2] += edges[edge]
        if node_edge_score[node_1] > max_score :#and node_1 not in ignore_nodes:
            max_score = node_edge_score[node_1]
            max_node = node_1
        if node_edge_score[node_2] > max_score :#and node_2 not in ignore_nodes:
            max_score = node_edge_score[node_2]
            max_node = node_2
    if max_node == -1:
        break
    #if max_node in selected_nodes:
        #print "Max node in selected nodes %s" %max_node
    assert max_node not in selected_nodes
    selected_nodes.add(max_node)
    to_be_deleted = []
    for edge in edges:
        if max_node == edge[0]:
            to_be_deleted.append(edge)
            ignore_nodes.add(edge[1])
            if edge[1] not in node_parents:
                node_parents[edge[1]] = set(edge[0])
            else:
                node_parents[edge[1]].add(edge[0])
        elif max_node == edge[1]:
            to_be_deleted.append(edge)
            ignore_nodes.add(edge[0])
            if edge[0] not in node_parents:
                node_parents[edge[0]] = set(edge[1])
            else:
                node_parents[edge[0]].add(edge[1])
    for edge in to_be_deleted:
        del edges[edge]
    print "Edge count reduced to %d" %len(edges)
    print "Number of nodes taken so far %d" %len(selected_nodes)
    print "Number of nodes ignored %d" %len(ignore_nodes)

print "Number of nodes left after thresholding %d" %len(nodes)
print "Number of nodes taken %d " %len(selected_nodes)
print "%d new edges from %d old edges" %(new_num_edges,original_num_edges)
selected_nodes_top_counts = {}
selected_nodes_important_counts = {}
answers = []
for node in selected_nodes:
    answers.append(node[:node.find('_')])
    if node in answer_top_counts:
        selected_nodes_top_counts[node] = answer_top_counts[node]
    else:
        selected_nodes_top_counts[node] = 0
    selected_nodes_important_counts[node] = interesting_answer_counts[node]
print selected_nodes_top_counts
print selected_nodes_important_counts
general_answers = open("general_answers.txt","w")
for answer in answers:
    index = bsl.solved_files.index(DATA_PATH+answer)
    question = bsl.solved_files_question_text[index]
    answer_text = bsl.solved_files_question_answer[index]+" "+bsl.map_support_docs_exact_title[bsl.solved_files_support_docs[DATA_PATH+answer]]
    url = "https://discussions.apple.com/thread/"+answer
    general_answers.write(url+"\nQuestion: "+question+"\nAnswer: "+answer_text+"\n\n\n")
general_answers.close()
#g = Graph(len(nodes))
#total_edges = 0
#edges_added = 0
#for edge in edges:
#   total_edges += 1
#   if edge_counts[edge] >EDGE_THRESHOLD:
#       g.addEdge(nodes.index(edge[0]),nodes.index(edge[1]))
#       edges_added += 1

#print "%d edges added of %d total edges" %(edges_added,total_edges)
#print ("Following are strongly connected components " +
#                           "in given graph")
#num_connected,components = g.printSCCs()
#print ""
#print ""
#print "%d edges added of %d total edges" %(edges_added,total_edges)
#print "Number of connected components are %d" %num_connected
#print "Number of original nodes are %d" %len(nodes)
#components_sizes = {}
#for component in components:
#    if len(component) in components_sizes:
#        components_sizes[len(component)] += 1
#    else:
#        components_sizes[len(component)] = 1
#print components_sizes
#assert len(dfs_list) == len(nodes)
#print np.mean(edges.values())
#print edges
#print difference_train
#print difference_count
#X = []
#Y = []
#for rank,count in difference_count.keys():
#    X.append(rank+1)
#    Y.append(difference_train[rank]*1.0/count)
#plt.plot(X,Y)
#plt.savefig('ranks.png')
