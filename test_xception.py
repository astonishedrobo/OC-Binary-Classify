import argparse
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Self Defined Modules
from utils.datasets import BioClassify
from model.models import XceptionClassifier

def inference(dataloader, model):
    probs = []
    labels = []
    for data, label in tqdm(dataloader):
        data = Variable(data.permute(0, 3, 1, 2).cuda())
        output = model(data)
        probs.extend(torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        labels.extend(label.numpy())
    return probs, labels

def test(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # Set model
    model = XceptionClassifier(num_classes=args.num_classes)
    checkpoint = torch.load(args.weights_path)
    model.load_state_dict(checkpoint)
    model = torch.compile(model)
    model.cuda()
    model.eval()

    # Load Dataset
    dataset = BioClassify(paths={"data": args.data_path, "target_stain": args.target_stain})
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Run Inference
    probs, labels = inference(dataloader, model)

    # Compute confusion matrix
    cm = confusion_matrix(labels, np.round(probs))
    class_names = [args.class_map[i] for i in range(args.num_classes)]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Add labels to each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f"./metrics/cm_{args.classifier}_{str(args.fold)}.png" if args.fold else f"./metrics/cm_{args.classifier}_test.png")

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./metrics/roc_curve_{args.classifier}_{str(args.fold)}.png" if args.fold else f"./metrics/roc_curve_{args.classifier}_test.png")

def parse_args():
    parser = argparse.ArgumentParser()
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    # Paths
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--target_stain", type=str, default="./data/images/target.png")
    parser.add_argument("--weights_path", type=str, required=True)
    # Miscellaneous
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--classifier", type=str, default=None)

    args = parser.parse_args()
    if args.classifier is not None:
        args.class_map = {0: args.classifier.split("_")[0], 1: args.classifier.split("_")[1]}
    else:
        args.class_map = None
    print(args.class_map)
    return args

if __name__ == "__main__":
    args = parse_args()
    test(args)
