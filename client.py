import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import tenseal as ts
import argparse
import logging
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

public_context = pickle.load(open("public_context.pkl", "rb"))
secret_context = pickle.load(open("secret_context.pkl", "rb"))

context = ts.context_from(secret_context)
context.make_context_public()
public_context = context.serialize()

# Check if a variable is a list or a np.ndarray
def isList(arrayList):
    if type(arrayList) == list or type(arrayList) == np.ndarray:
        return True
    else:
        return False

# Encrypt data function
def cipher(values, context):
    tempValues = []
    for i in range(len(values)):
        if isList(values[i]):
            tempValues.append(cipher(values[i], context))
        else:
            encrypt = ts.ckks_vector(context, values)
            encrypt = encrypt.serialize()
            return encrypt
    return tempValues

# Decrypt data function
def plain(values, context):
    tempValues = values
    for i in range(len(values)):
        if isList(values[i]):
            plain(values[i], context)
        else:
            if type(values[i]) == ts.tensors.ckksvector.CKKSVector:
                values[i].link_context(context)
                tempValues[i] = values[i].decrypt()
    return tempValues

# Deserialize data to get bytes
def deserializeToBytes(values):
    tempValues = []
    for i in range(len(values)):
        if isList(values[i]):
            if len(values[i].shape) == 0:
                tempValues.append(values[i].tobytes())
            else:
                tempValues.append(deserializeToBytes(values[i]))
        else:
            tmp = []
            for elem in values:
                tmp.append(elem.tobytes())
            return tmp
    return tempValues

# Deserialize data function
def deserialize(values, context):
    tempValues = values
    for i in range(len(values)):
        if isList(values[i]):
            deserialize(values[i], context)
        elif type(values[i]) == bytes:
            deser = ts.ckks_vector_from(context, values[i])
            tempValues[i] = deser
    return tempValues

# Evaluate model
def evaluate_model(model: torch.nn.Module, testloader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = running_loss / len(testloader)
    return loss, accuracy

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, testloader, net, criterion, optimizer) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

    def get_parameters(self, config):
        logging.debug(f"[Client {self.cid}] get_parameters")
        ndarrays = [param.cpu().detach().numpy() for param in self.net.parameters()]
        encryptedParameters = cipher(ndarrays, ts.context_from(public_context))
        return encryptedParameters

    def set_parameters(self, parameters: Parameters) -> None:
        logging.debug(f"[Client {self.cid}] set_parameters")
        deserializedBytesParameters = deserializeToBytes(parameters)
        deserializedParams = deserialize(deserializedBytesParameters, ts.context_from(public_context))
        private_context = ts.context_from(secret_context)
        decryptedParams = plain(deserializedParams, private_context)
        params_dict = zip(self.net.state_dict().keys(), decryptedParams)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict)

    def fit(self, parameters, config) -> fl.common.FitRes:
        logging.debug(f"[Client {self.cid}] fit")
        self.set_parameters(parameters)
        self.train_local(self.net, self.trainloader, 1, self.criterion, self.optimizer, torch.device("cpu"))
        ndarrays_updated = [param.cpu().detach().numpy() for param in self.net.parameters()]
        encryptedParameters = cipher(ndarrays_updated, ts.context_from(public_context))
        return encryptedParameters, len(self.trainloader.dataset), {}

    def evaluate(self, parameters: Parameters, config) -> fl.common.EvaluateRes:
        logging.debug(f"[Client {self.cid}] Starting evaluation.")
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(self.net, self.testloader, self.criterion, torch.device("cpu"))
        logging.debug(f"[Client {self.cid}] Evaluation completed. Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(self.testloader.dataset), {}

    def train_local(self, model, trainloader, epochs, criterion, optimizer, device):
        model.train()
        for epoch in range(epochs):
            for data, labels in trainloader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="ID of the partition to use")
    args = parser.parse_args()

    # Load dataset and create dataloaders
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    trainset_partitions = torch.utils.data.random_split(mnist_train, [len(mnist_train) // 5] * 5)
    trainloaders = [DataLoader(partition, batch_size=16, shuffle=True) for partition in trainset_partitions]
    testloader = DataLoader(mnist_test, batch_size=16, shuffle=False)

    # Create model, criterion, and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Start the Flower client
    fl.client.start_client(server_address="localhost:8080", client=fl.client.NumPyClient.to_client(FlowerClient(str(args.partition_id), trainloaders[args.partition_id], testloader, net, criterion, optimizer)))
