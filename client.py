import argparse
import tenseal as ts
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pickle

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)  # Adjusted input size for MNIST
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*7*7)  # Adjusted input size for MNIST
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    partition_size = len(train_dataset) // 5
    partitions = random_split(train_dataset, [partition_size]*5)
    
    return partitions, test_dataset

def encrypt_parameters(parameters, context):
    encrypted_parameters = []
    for param in parameters:
        param_flat = param.flatten().tolist()  # Flatten the tensor to a list
        encrypted_tensor = ts.ckks_vector(context, param_flat)
        encrypted_parameters.append((encrypted_tensor.serialize(), param.shape))  # Store the shape for later use
    return encrypted_parameters

def decrypt_parameters(encrypted_parameters, context):
    decrypted_parameters = []
    for enc_param, shape in encrypted_parameters:
        enc_param = ts.ckks_vector_from(context, enc_param)
        decrypted_tensor = torch.tensor(enc_param.decrypt()).reshape(shape)  # Reshape to the original shape
        decrypted_parameters.append(decrypted_tensor)
    return decrypted_parameters

def serialize_parameters(parameters):
    return pickle.dumps(parameters)

def deserialize_parameters(serialized_parameters):
    return pickle.loads(serialized_parameters)

def client_fn(partition_id):
    partitions, _ = load_datasets()
    train_loader = DataLoader(partitions[partition_id], batch_size=32, shuffle=True)

    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Initialize TenSEAL context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            encrypted_params = encrypt_parameters(parameters, context)
            serialized_params = serialize_parameters(encrypted_params)
            return [np.frombuffer(serialized_params, dtype=np.uint8)]

        def set_parameters(self, parameters):
            serialized_params = parameters[0].tobytes()
            encrypted_params = deserialize_parameters(serialized_params)
            decrypted_params = decrypt_parameters(encrypted_params, context)
            params_dict = zip(model.state_dict().keys(), decrypted_params)
            state_dict = {k: v.clone().detach() for k, v in params_dict}  # Updated line
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            for epoch in range(3):  # Aumenta il numero di epoche a 3
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
            return self.get_parameters(), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            _, test_dataset = load_datasets()
            test_loader = DataLoader(test_dataset, batch_size=32)
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(test_loader.dataset)
            return float(accuracy), len(test_loader.dataset), {"accuracy": accuracy}

    return FlowerClient()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--partition-id", type=int, required=True, help="Partition ID")
    args = parser.parse_args()

    fl.client.start_client(
        server_address="localhost:8080",
        client=client_fn(args.partition_id).to_client(),
    )
