import subprocess
import os
import argparse

# Function to train a single fold
def train_fold(train_data, valid_data, fold_num, script_path, classifier):
    command = f"python3 {script_path} --train_data={train_data} --valid_data={valid_data} --fold={fold_num} --classifier={classifier}"
    subprocess.run(command, shell=True)

def train_folds(data_path, script_path, classifier, num_folds=3):
    fold_paths_tain = [f"{data_path}/fold_{fold}_train.csv" for fold in range(1, num_folds + 1)]
    fold_paths_valid = [f"{data_path}/fold_{fold}_val.csv" for fold in range(1, num_folds + 1)]

    for fold, (fold_train, fold_valid) in enumerate(zip(fold_paths_tain, fold_paths_valid)):
        train_data = fold_train 
        valid_data = fold_valid

        print(f"Training Fold {fold + 1}")
        # command = f"python3 {os.path(script_path)} --train_data={os.path(train_data)} --valid_data={os.path(valid_data)} --fold={fold + 1}"
        # subprocess.run(command, shell=True)
        train_fold(train_data, valid_data, fold + 1, script_path, classifier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on each fold")
    parser.add_argument("--classifier", type=str, required=True, help="Classifier to train")
    parser.add_argument("--script_path", type=str)
    args = parser.parse_args()

    data_path = f"./data/folds_data_{args.classifier}"
    script_path = "./train_xception.py"
    classifier = args.classifier
    train_folds(data_path, script_path, classifier)
