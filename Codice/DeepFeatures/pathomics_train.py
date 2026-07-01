import datetime
import os
import time
from sys import platform

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lib import networks, utils
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision.models as pymodels


def training_loop(n_epochs, optimizer, net, loss_fn, train_loader, val_loader, device, writer):
    writer = writer
    epochs_idx = 0
    ts_epoch = 0
    ts = time.time()
    for epoch in range(1, n_epochs + 1):
        epochs_idx += 1
        loss_train = 0.0
        net.train()
        # Iterate over data.
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 1 == 0:
            speed = (epochs_idx - ts_epoch) / (time.time() - ts)
            ts_epoch = epochs_idx
            ts = time.time()
            val_loss = valid_loss(net, loss_fn, val_loader, device)
            train_loss = loss_train / len(train_loader)
            train_acc = model_accuracy(net, train_loader, device)
            val_acc = model_accuracy(net, val_loader, device)
            print('{} Epoch {}/{}, Training loss {: .4f}, Training accuracy {: .4f}, Validation loss {: .4f}, Validation accuracy {: .4f}, Speed {: .5f} epochs/s'.format(
                datetime.datetime.now(), epoch, n_epochs, train_loss, train_acc, val_loss, val_acc, speed))

            writer.add_scalar("Loss/Train", train_loss, epochs_idx)
            writer.add_scalar("Accuracy/Train", train_acc, epochs_idx)
            writer.add_scalar("Loss/Val", val_loss, epochs_idx)
            writer.add_scalar("Accuracy/Val", val_acc, epochs_idx)
            writer.add_scalar("Speed", speed, epochs_idx)


def valid_loss(net, loss_fn, val_loader, device):
    test_loss = 0
    net.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
    return test_loss / len(val_loader)


def model_accuracy(net, loader, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = net(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
    return correct / total


if __name__ == '__main__':
    if platform == "linux" or platform == "linux2":
        csv_file = "data/pathomics/dataset_sliding.csv"
        root_dir = "data/pathomics/sliding_window"
        id_patient_path = "data/pathomics/crop_label.csv"
    elif platform == "win32":
        csv_file = r"data/pathomics/dataset_sliding.csv"
        id_patient_path = r"data/pathomics/crop_label.csv"
        root_dir = r"data/pathomics/sliding_window"
    elif platform == "darwin":
        csv_file = "data/pathomics/dataset_sliding.csv"
        id_patient_path = "data/pathomics/crop_label.csv"
        root_dir = "data/pathomics/sliding_window"

    results_path = "results"

    test_number = input(f"insert tensorboard results folder: ")
    if not os.path.exists(os.path.join(results_path, test_number)):
        os.makedirs(os.path.join(results_path, test_number))
    else:
        test_number = input(
            f"{test_number} already exists, insert new folder: ")
        os.mkdir(os.path.join(results_path, test_number))


    id_patients = pd.read_csv(csv_file, sep=",")
    id_patients = id_patients["file_name"].str.split("/", expand=True)[
        0].unique()

    n_splits = input("insert number k of folds: ")
    if n_splits == "loo": 
        n_splits = len(id_patients)
    else:
        n_splits = int(n_splits)

    n_epochs = 50
    num_classes = 2
    path_dim = 32


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}.")



    model_ft = networks.get_model("vgg19", num_classes=num_classes, path_dim=path_dim, mode = "classification")
    model_ft.to(device)

    summary(model_ft, (3, 224, 224))

    i = 0
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(id_patients):
        i += 1
        utils.append_list_as_row(os.path.join(results_path, test_number,
                                              "k-fold_testdata.csv"), [f"folder-{i}"] + id_patients[test_index].tolist())
    
    i = 0
    for train_index, test_index in kf.split(id_patients):
        i += 1

        writer = SummaryWriter(os.path.join(results_path, test_number, "runs"))

        train_loader, valid_loader = utils.split_dataloader(
            csv_file, root_dir, id_patients, train_index, test_index)

        optimizer = optim.SGD(model_ft.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        print('{} Experiment {} of {}'.format(
            datetime.datetime.now(), i, n_splits))
        training_loop(n_epochs=n_epochs, optimizer=optimizer, net=model_ft, loss_fn=loss_fn,
                      train_loader=train_loader, val_loader=valid_loader, device=device, writer=writer)

        nets_path = os.path.join(results_path, test_number, "nets")
        if not os.path.exists(nets_path):
            os.makedirs(nets_path)
        torch.save(model_ft.state_dict(), os.path.join(
            nets_path, f"net_{i}-fold.dat"))


        # Azzero rete per il prossimo paziente di test
        model_ft = networks.get_model("vgg19", num_classes=num_classes, path_dim=path_dim, mode = "classification")
        model_ft.to(device)
        writer.close()
