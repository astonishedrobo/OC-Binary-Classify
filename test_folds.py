import subprocess
import os
import argparse

# Function to test a single fold
def test_fold(test_data, fold_num, script_path, classifier, weight_path):
    if fold_num:
        command = f"python3 {script_path} --data_path={test_data} --fold={fold_num} --classifier={classifier} --weights_path={weight_path}"
        print(command)
    else: 
        command = f"python3 {script_path} --data_path={test_data} --classifier={classifier} --weights_path={weight_path}"
    subprocess.run(command, shell=True)

def test_folds(data_path, script_path, classifier, weights_path, train_metrics=False, num_folds=3):
    if train_metrics:
        fold_paths_tain = [f"{data_path}/fold_{fold}_train.csv" for fold in range(1, num_folds + 1)]
        fold_paths_valid = [f"{data_path}/fold_{fold}_val.csv" for fold in range(1, num_folds + 1)]
    else:
        fold_paths_test = [f"{data_path}/test.csv"]

    if train_metrics:
        for fold, (fold_train, fold_valid, weight_path) in enumerate(zip(fold_paths_tain, fold_paths_valid, weights_path)):
            train_data = fold_train 
            valid_data = fold_valid
            print(f"Testing Fold {fold + 1}")
            # test_fold(train_data, fold + 1, script_path, classifier, weight_path)
            test_fold(valid_data, fold + 1, script_path, classifier, weight_path)
    else:
        for fold_test, weight_path in zip(fold_paths_test, weights_path):  
            test_data = fold_test 
            print("Testing for Test Set")
            test_fold(test_data, None, script_path, classifier, weight_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on each fold")
    parser.add_argument("--classifier", type=str, required=True, help="Classifier to test")
    parser.add_argument("--script_path", type=str)
    parser.add_argument("--train_metrics", type=bool, default=False)
    parser.add_argument("--weights_path", nargs='+', type=str, required=True, help="Path to the trained weights for each fold")
    args = parser.parse_args()
    # args.weights_path = list(args.weights_path)
    print(args.weights_path)

    if args.train_metrics:
        data_path = f"./data/folds_data_{args.classifier}" 
    else:
        data_path = f"./data/folds_data_{args.classifier}_test" 
    script_path = "./test_inception.py"
    classifier = args.classifier
    weights_path = [f"./checkpoints/{weight_path}" for weight_path in args.weights_path]

    test_folds(data_path, script_path, classifier, weights_path, args.train_metrics)