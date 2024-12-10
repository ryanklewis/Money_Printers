from models.mlp import MLP
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path


def get_data_loaders(device, label="label_1", batch_size=64, use_custom_cols=True, sliding_window=1):
    train_data = pd.read_csv("data/Train_NoAuction_Zscore.csv")
    test_data = pd.read_csv("data/Test_NoAuction_Zscore.csv")

    X_train = train_data.drop(
        columns=["label_1", "label_2", "label_3", "label_5", "label_10"])
    X_test = test_data.drop(
        columns=["label_1", "label_2", "label_3", "label_5", "label_10"])

    y_train = train_data[label] - 1
    y_test = test_data[label] - 1
    if not use_custom_cols:
        X_train = X_train.iloc[:, :40]
        X_test = X_test.iloc[:, :40]

    # sliding window stuff

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, shuffle=True, test_size=0.2)

    X_train_tensor = torch.tensor(
        X_train.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(
        X_val.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(
        X_test.values, dtype=torch.float32).to(device)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def test_model(model, test_loader, experiment_name, save=True, display=False):
    test_predictions = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
    report = classification_report(test_labels, test_predictions)
    if display:
        print(report)
    if save:
        with open(f"results/{experiment_name}/report.txt", "w") as f:
            f.write(report)


def visualize_curves(train_losses, val_losses, train_accuracies, val_accuracies, experiment_name, save=True, display=False):
    plt.plot(np.arange(epochs), train_losses, label="train")
    plt.plot(np.arange(epochs), val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curves")
    plt.legend()
    if display:
        plt.show()
    if save:
        plt.savefig(f"results/{experiment_name}/loss_curves.png")
    plt.close()

    plt.plot(np.arange(epochs), train_accuracies, label="train")
    plt.plot(np.arange(epochs), val_accuracies, label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy curves")
    plt.legend()
    if display:
        plt.show()
    if save:
        plt.savefig(f"results/{experiment_name}/accuracy_curves.png")


def train_and_evaluate_model(train_loader, val_loader, test_loader, experiment_name, optimizer, criterion, epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for i in range(epochs):
        model.train()
        running_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        for batch, (X, y) in enumerate(train_loader):
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_train_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            train_correct_predictions += (predicted == y).sum().item()
            train_total_samples += y.size(0)

        print(f"Epoch {i} train loss {running_train_loss/len(train_loader)}")
        train_losses.append(running_train_loss/len(train_loader))

        train_accuracy = train_correct_predictions / train_total_samples
        train_accuracies.append(train_accuracy)
        print(f"Epoch {i} train accuracy {train_accuracy}")

        val_correct_predictions = 0
        val_total_samples = 0
        model.eval()
        with torch.no_grad():
            running_val_loss = 0
            for batch, (X, y) in enumerate(val_loader):
                pred = model(X)
                loss = criterion(pred, y)
                running_val_loss += loss.item()

                _, predicted = torch.max(pred, 1)
                val_correct_predictions += (predicted == y).sum().item()
                val_total_samples += y.size(0)

            print(f"Epoch {i} val loss {running_val_loss/len(val_loader)}")
            val_losses.append(running_val_loss/len(val_loader))

            val_accuracy = val_correct_predictions / val_total_samples
            val_accuracies.append(val_accuracy)
            print(f"Epoch {i} val accuracy {val_accuracy}")

    test_model(model, test_loader, experiment_name)
    visualize_curves(train_losses, val_losses,
                     train_accuracies, val_accuracies, experiment_name)


if __name__ == "__main__":
    # get device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 2

    # get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        device, batch_size=64)

    # model definition
    model = MLP(input_size=144, hidden_size=64)
    model = model.to(device)

    # experiment name
    experiment_name = "test"
    Path(f"results/{experiment_name}").mkdir(parents=True, exist_ok=True)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training and evaluation
    train_and_evaluate_model(train_loader, val_loader,
                             test_loader, "test", optimizer, criterion, epochs)
