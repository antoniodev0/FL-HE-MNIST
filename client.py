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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
# Carica il contesto TenSEAL
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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
        train(self.net, self.trainloader, epochs=5)
        # Genera la confusion matrix alla fine del training
        self.generate_confusion_matrix()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"partition_id": self.cid}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, accuracy = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss), "accuracy": float(accuracy)}
    def generate_confusion_matrix(self):
        """Genera e salva la matrice di confusione per questo client."""
        all_preds = []
        all_labels = []

        # Ottieni le predizioni e le etichette vere
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.valloader:
                outputs = self.net(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcola la confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Visualizza la confusion matrix senza normalizzazione
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')  # Usa 'fmt="d"' per valori interi
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix Client {self.cid} (Absolute Counts)')

        # Salva l'immagine con il nome del client
        plt.savefig(f'confusion_matrix_client_{self.cid}.png')
        plt.close()

def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return val_loss / len(testloader), accuracy

def load_data(partition_id, num_partitions=20):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Partiziona il dataset in 30 parti
    partition_size = len(mnist_train) // num_partitions
    partitions = random_split(mnist_train, [partition_size] * num_partitions)
    
    # Seleziona la partizione per questo client
    partition = partitions[partition_id]
    
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
