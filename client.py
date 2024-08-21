import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import tenseal as ts
import pickle
import argparse

# Carica il contesto TenSEAL
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)


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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HomomorphicFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, testloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_parameters(self, config):
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]
        encrypted_params = [ts.ckks_vector(context, param.flatten()).serialize() for param in params]
        return encrypted_params

    def set_parameters(self, parameters):
        params = []
        for param in parameters:
            # Converte l'array NumPy in un buffer di byte
            serialized_param = param.tobytes()
            # Usa lazy_ckks_vector_from con il buffer serializzato
            ckks_vector = ts.lazy_ckks_vector_from(serialized_param)
            # Link il contesto al vettore CKKS deserializzato
            ckks_vector.link_context(context)
            # Decifra il vettore CKKS utilizzando il contesto con la chiave segreta
            decrypted_param = ckks_vector.decrypt()
            # Converti la lista decifrata in un array NumPy
            decrypted_param = np.array(decrypted_param)
            params.append(decrypted_param)

        # Ricostruisci lo state_dict
        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = {k: torch.Tensor(v.reshape(self.net.state_dict()[k].shape)) for k, v in params_dict}
        
        # Carica i parametri nel modello
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss)}

def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
    return val_loss / len(testloader)

def load_data(partition_id):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Partiziona il dataset in base a partition_id
    n_partitions = 5
    partition_size = len(mnist_train) // n_partitions
    partition = random_split(mnist_train, [partition_size] * n_partitions)[partition_id]
    
    # Dividi in train (80%) e validation (20%)
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_subset, val_subset = random_split(partition, [train_size, val_size])
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=32)
    testloader = DataLoader(mnist_test, batch_size=32)
    
    return trainloader, valloader, testloader

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="ID of the partition to use")
    args = parser.parse_args()

    trainloader, valloader, testloader = load_data(args.partition_id)
    
    net = Net()
    
    fl.client.start_client(
        server_address="localhost:8080",
        client=HomomorphicFlowerClient(str(args.partition_id), net, trainloader, valloader, testloader).to_client()
    )

if __name__ == "__main__":
    main()
