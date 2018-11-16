import pickle
import numpy as np
from sklearn.cluster.bicluster import SpectralBiclustering

question_topics = pickle.load(open("question_topics.p","r"))
answer_topics = pickle.load(open("answer_topics.p","r"))
test_question_topics = pickle.load(open("test_question_topics.p","r"))

cluster_input = None
for ind,question in enumerate(question_topics.keys()):
    assert question in answer_topics
    if cluster_input == None:
        cluster_input = np.outer(answer_topics[question],question_topics[question])
    else:
        cluster_input += np.outer(answer_topics[question],question_topics[question])
    assert cluster_input.shape[0] == np.array(answer_topics[question]).shape[0]
    assert cluster_input.shape[1] == np.array(question_topics[question]).shape[0]
    if ind%1000 == 0:
        print ind

print "Number of answer topics %d" %np.array(answer_topics[question]).shape[0]
print "Number of question topics %d" %np.array(question_topics[question]).shape[0]

cluster_input /= len(question_topics)
n_clusters = (10,10)
model = SpectralBiclustering(n_clusters=n_clusters,n_init=1000,mini_batch=False)
model.fit(cluster_input)
answer_clusters = model.row_labels_
question_clusters = model.column_labels_

values_map = {}
for i in range(cluster_input.shape[0]):
    for j in range(cluster_input.shape[1]):
        if tuple([model.row_labels_[i],model.column_labels_[j]]) not in values_map:
            values_map[tuple([model.row_labels_[i],model.column_labels_[j]])] = cluster_input[i][j]
        else:
            values_map[tuple([model.row_labels_[i],model.column_labels_[j]])] += cluster_input[i][j]

print values_map

f = file("question_clusters.bin","wb")
np.save(f,np.array(question_clusters))
f.close()

f = file("answer_clusters.bin","wb")
np.save(f,np.array(answer_clusters))
f.close()

original_question_clusters = {}
original_answer_clusters = {}
merged_question_clusters = {}
merged_answer_clusters = {}

merged_question_answer_lists = {}

for ind,question in enumerate(question_topics.keys()):
    question_topic = np.array(question_topics[question]).argmax()
    #merged_question_topic = question_clusters[question_topic]
    merged_question_topic = question_topic
    if merged_question_topic not in merged_question_clusters:
        merged_question_clusters[merged_question_topic] = [question]
    else:
        merged_question_clusters[merged_question_topic].append(question)
    
    answer_topic = np.array(answer_topics[question]).argmax()
    #merged_answer_topic = answer_clusters[answer_topic]
    merged_answer_topic = answer_topic
    if merged_answer_topic not in merged_answer_clusters:
        merged_answer_clusters[merged_answer_topic] = [question]
    else:
        merged_answer_clusters[merged_answer_topic].append(question)

    if merged_question_topic not in merged_question_answer_lists:
        merged_question_answer_lists[merged_question_topic] = [merged_answer_topic]
    else:
        merged_question_answer_lists[merged_question_topic].append(merged_answer_topic)

count = 0
total = 0
from collections import Counter

for topic in merged_question_answer_lists:
    answer_topicz = merged_question_answer_lists[topic]
    most_common,num_most_common = Counter(answer_topicz).most_common(1)[0]
    count += num_most_common
    total += len(answer_topicz)

print "Fraction of those that count %f" %((1.0*count)/total)



for test_question in question_topics:
    assert test_question  in question_topics
    test_question_topic_distribution = question_topics[test_question]
    topic = np.array(test_question_topic_distribution).argmax()
    mega_topic = model.column_labels_[topic]
    mega_answer_topic = -1
    mega_answer_topic_score = 0.0
    for i in model.row_labels_:
        current_value = values_map[tuple([i,mega_topic])]
        if current_value > mega_answer_topic_score:
            mega_answer_topic = i
            mega_answer_topic_score = current_value
    answer_topics_found = set()
    for i in range(len(model.row_labels_)):
        if model.row_labels_[i] == mega_answer_topic:
            answer_topics_found.add(i)
    questions_for_those_answers = []
    for question in answer_topics.keys():
        if np.array(answer_topics[question]).argmax() in answer_topics_found and question in question_topics:
            questions_for_those_answers.append(question)
    print "Number of candidate questions to compare = %d" %len(questions_for_those_answers)
    if test_question in questions_for_those_answers:
        print "Found the right slot"
    else:
        print "Found the incorrect slot"
    best_question = ""
    best_question_score = 0.0
    for question in questions_for_those_answers:
        current_value = np.dot(np.array(question_topics[question]),np.array(test_question_topic_distribution))
        current_value /= np.linalg.norm(np.array(question_topics[question]))
        current_value /= np.linalg.norm(np.array(test_question_topic_distribution))
        if current_value > best_question_score:
            best_question_score = current_value
            best_question = question
    print "Best question for %s is %s" %(test_question,best_question)
    print "Best score = %f" %best_question_score
            
#possible_answer_clusters = {}
#test_nearest_neigbour = {}
#for test_question in test_question_topics:
#    question_topicz = test_question_topics[test_question]
#    topic = np.array(question_topicz).argmax()
#    mega_topic = model.column_labels_[topic]
#    print "Mega topic found %d" %mega_topic
#    mega_answer_topic = -1
#    mega_answer_value = 0.0
#    for i in range(n_clusters[0]):
#        current_value = values_map[tuple([i,mega_topic])]
#        print current_value,mega_answer_value
#        if current_value > mega_answer_value:
#            mega_answer_topic = i
#            mega_answer_value = current_value
#    answer_clusters = []
#    print "Mega answer topic found %d" %mega_answer_topic
#    for i in range(len(model.row_labels_)):
#        if model.row_labels_[i] == mega_answer_topic:
#            answer_clusters.append(i)
#    possible_answer_clusters[test_question] = answer_clusters
#    print len(answer_clusters)
#    print test_question
#    print answer_clusters
#    best_question = ""
#    best_question_score = 0.
#    for question in question_topics:
#        train_answer_topics = answer_topics[question]
#        train_answer_topic = np.array(train_answer_topics).argmax()
#        if train_answer_topic not in answer_clusters:
#            continue
#        train_question_topics = question_topics[question]
#        current_question_score = np.dot(np.array(question_topics[question]),np.array(question_topicz))/(np.linalg.norm(np.array(question_topics[question]))*np.linalg.norm(np.array(question_topicz)))
#        if current_question_score > best_question_score:
#            best_question = question
#            best_question_score  = current_question_score
#    test_nearest_neigbour[test_question] = best_question

#answer_cluster_distributions = {}
#for i in range(len(model.row_labels_)):
#    if model.row_labels_[i] not in answer_cluster_distributions:
#        answer_cluster_distributions[model.row_labels_[i]] = 1
#    else:
#        answer_cluster_distributions[model.row_labels_[i]] += 1
#print answer_cluster_distributions
#print test_nearest_neigbour
