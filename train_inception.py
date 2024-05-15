import argparse
import torch
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import nn

# Import Self Defined Modules
from utils.datasets import BioClassify
from model.models import *

def train(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # Set datasets
    train_dataset = BioClassify(paths={"data": args.train_data, "target_stain": args.target_stain})
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set model
    model = InceptionV3Classifier(num_classes=args.num_classes, pretrained=args.use_imagenet_weights)  # Change to your classification model
    # model = torch.compile(model)
    model.cuda()
    

    # Set loss function
    criterion = nn.functional.cross_entropy  # Change to your classification loss function

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set Paths
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set tensorboard writer
    run_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, "loss", f"{run_time}_{tuple(model._modules)[-1]}_{args.classifier}_{str(args.fold)}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Track Metrics
    train_loss = []
    valid_loss = []

    # Train
    for epoch in range(args.epochs):
        model.train()
        loss_meter = []
        for data, label in tqdm(train_loader):
            data = data.permute(0, 3, 1, 2)
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            if epoch < args.warmup_epochs and args.use_imagenet_weights:
                # Freeze all layers except the final fc layer
                for name, param in model.named_parameters():
                    if name != f"{tuple(model._modules)[-1]}.fc.weight" and name != f"{tuple(model._modules)[-1]}.fc.bias":
                        param.requires_grad = False
                output = torch.as_tensor(model(data)[0])
            else:
                if epoch == args.warmup_epochs and args.use_imagenet_weights:
                    # Unfreeze all layers
                    print("Start: Training the whole network")
                    for param in model.parameters():
                        param.requires_grad = True
                    args.use_imagenet_weights = False
                output = torch.as_tensor(model(data)[0])

            loss = criterion(output, label)
            loss_meter.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_avg = np.mean(loss_meter)
        train_loss.append(loss_avg)
        writer.add_scalar("Loss/train", loss_avg, epoch)
        print(f"(Train) Epoch: {epoch+1}/{args.epochs} Loss: {loss_avg}")

        if epoch % args.save_every == 0:
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(args.save_path, f"{run_time}_{tuple(model._modules)[-1]}_{args.classifier}_{str(args.fold)}_model.pth"))

            # Validate (for classification task)
            if args.valid_data:
                val_loss = validate(args, model, criterion)
                valid_loss.append(val_loss)
                print(f"(Valid) Epoch: {epoch+1}/{args.epochs} Loss: {val_loss}")
                writer.add_scalar("Loss/valid", val_loss, epoch)

                # Save best model based on accuracy
                if val_loss == np.min(valid_loss):
                    torch.save(model.state_dict(), os.path.join(args.save_path, f"{run_time}_{tuple(model._modules)[-1]}_{args.classifier}_{str(args.fold)}_model_best.pth"))

def validate(args, model, criterion):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set datasets
    valid_dataset = BioClassify(paths={"data": args.valid_data, "target_stain": args.target_stain})
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Run validation
    model.eval()
    loss_meter = []
    with torch.no_grad():
        for data, label in tqdm(valid_loader):
            data = data.permute(0, 3, 1, 2)
            data = Variable(data.to(device))
            label = Variable(label.to(device))
            output = torch.as_tensor(model(data))
            loss = criterion(output, label)
            loss_meter.append(loss.item())
    
    return np.mean(loss_meter)

def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--save_every", type = int, default = 2)
    parser.add_argument("--num_workers", type = int, default = 5)

    # Paths
    parser.add_argument("--train_data", type = str, default = "")
    parser.add_argument("--valid_data", type = str, default = None)
    parser.add_argument("--target_stain", type = str, default = "./data/images/target.png")
    parser.add_argument("--save_path", type = str, default = "./checkpoints/")
    parser.add_argument("--log_dir", type = str, default = "./logs/")

    # Miscellanous
    parser.add_argument("--num_classes", type = int, default = 2)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--use_imagenet_weights", type=int, default=True)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    args = parser.parse_args()
    assert args.epochs > args.warmup_epochs, "Epochs must be greater than Warmup Epochs"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
