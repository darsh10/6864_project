print "Hello Darsh, I am now beginning the run of the pipeline"
print "Beginning the initial run of data generation"
from subprocess import call
call(["rm","model.pkl.gz"])
call(["rm","ans_encoding.p"])
call(["rm","/scratch/darsh/dialogue_systems/tao_model/train.txt"])
call(["rm","/scratch/darsh/dialogue_systems/tao_model/test.txt"])
call(["rm","/scratch/darsh/dialogue_systems/tao_model/tao_model.txt"])
call(["rm","/scratch/darsh/dialogue_systems/tao_model/train_cosine.txt"])
call(["rm","train.txt"])
call(["rm","test.txt"])
call(["rm","train_cosine.txt"])
call(["rm","tao_model.txt"])
call(["time","python","baseline_new.py"])
call(["cp","train.txt","test.txt","tao_model.txt","train_cosine.txt","/scratch/darsh/dialogue_systems/tao_model/"])
call(["python","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/main.py","--corpus","/scratch/darsh/dialogue_systems/tao_model/tao_model.txt","--embeddings","/scratch/darsh/dialogue_systems/tao_model/apple_question_answer_vectors.txt","--train","/scratch/darsh/dialogue_systems/tao_model/train.txt","--test","/scratch/darsh/dialogue_systems/tao_model/test.txt","--dropout","0.1","-d","400","--save_model","model.pkl.gz","--max_epoch","10"])
call(["python","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/main_cosine.py","--corpus","/scratch/darsh/dialogue_systems/tao_model/tao_model.txt","--embeddings","/scratch/darsh/dialogue_systems/tao_model/apple_question_answer_vectors.txt","--train","/scratch/darsh/dialogue_systems/tao_model/train_cosine.txt","--load_pretrain","model.pkl.gz","-d","400"])
call(["mv","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/ans_encoding.p","./"])
call(["time","python","baseline_new.py","1"])

for i in range(5):
    call(["cp","train.txt","/scratch/darsh/dialogue_systems/tao_model/"])
    print "Running epoch %d" %(i+1)
    call(["python","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/main.py","--corpus","/scratch/darsh/dialogue_systems/tao_model/tao_model.txt","--embeddings","/scratch/darsh/dialogue_systems/tao_model/apple_question_answer_vectors.txt","--train","/scratch/darsh/dialogue_systems/tao_model/train.txt","--test","/scratch/darsh/dialogue_systems/tao_model/test.txt","--dropout","0.1","-d","400","--save_model","model.pkl.gz","--max_epoch","5","--load_pretrain","model.pkl.gz"])
    call(["python","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/main_cosine.py","--corpus","/scratch/darsh/dialogue_systems/tao_model/tao_model.txt","--embeddings","/scratch/darsh/dialogue_systems/tao_model/apple_question_answer_vectors.txt","--train","/scratch/darsh/dialogue_systems/tao_model/train_cosine.txt","--load_pretrain","model.pkl.gz","-d","400"])
    call(["mv","/scratch/darsh/dialogue_systems/tao_model/rcnn/code/qa/ans_encoding.p","./"])
    call(["time","python","baseline_new.py","1"])
