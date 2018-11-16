python svm_rank.py -t 0.75 -r svm/qa_train.dat -s svm/qa_test.dat
./svm/svm_rank_learn -c 3 svm/qa_train.dat svm/qa_model
./svm/svm_rank_classify svm/qa_test.dat svm/qa_model svm/qa_predictions
python svm_rank_scoring.py -t 1 -p svm/qa_predictions -d svm/qa_test.dat

# python svm_rank.py -t 0.75 -r svm/qa_train_binary.dat -s svm/qa_test_binary.dat -b
# ./svm/svm_rank_learn -c 3 svm/qa_train_binary.dat svm/qa_model_binary
# ./svm/svm_rank_classify svm/qa_train_binary.dat svm/qa_model_binary svm/qa_predictions_binary
# python svm_rank_scoring.py -t 20 -p svm/qa_predictions_binary -d svm/qa_train_binary.dat

# ./svm/svm_rank_learn -c 3 svm/example3/train.dat svm/example3/model 
# ./svm/svm_rank_classify svm/example3/test.dat svm/example3/model svm/example3/predictions
# python svm_rank_scoring.py -t 1 -p svm/example3/predictions -d svm/example3/test.dat