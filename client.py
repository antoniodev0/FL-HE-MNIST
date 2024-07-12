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
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
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

def encrypt_parameters(parameters, context, batch_size=2048):
    encrypted_parameters = []
    for param in tqdm(parameters, desc="Encrypting Parameters"):
        param_flat = param.flatten().tolist()
        encrypted_batches = []
        for i in range(0, len(param_flat), batch_size):
            batch = param_flat[i:i+batch_size]
            encrypted_tensor = ts.ckks_vector(context, batch)
            encrypted_batches.append(encrypted_tensor.serialize())
        encrypted_parameters.append((encrypted_batches, param.shape))
    return encrypted_parameters

def decrypt_parameters(encrypted_parameters, context, batch_size=2048):
    decrypted_parameters = []
    for enc_batches, shape in tqdm(encrypted_parameters, desc="Decrypting Parameters"):
        decrypted_flat = []
        for enc_batch in enc_batches:
            enc_tensor = ts.ckks_vector_from(context, enc_batch)
            decrypted_flat.extend(enc_tensor.decrypt())
        decrypted_tensor = torch.tensor(decrypted_flat).reshape(shape)
        decrypted_parameters.append(decrypted_tensor)
    return decrypted_parameters

def serialize_parameters(encrypted_parameters):
    return pickle.dumps(encrypted_parameters)

def deserialize_parameters(serialized_parameters):
    return pickle.loads(serialized_parameters)

def client_fn(partition_id):
    partitions, _ = load_datasets()
    train_loader = DataLoader(partitions[partition_id], batch_size=16, shuffle=True)  # Reduced batch size

    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            parameters = [val.cpu().numpy() for val in model.state_dict().values()]
            encrypted_params = encrypt_parameters(parameters, context)
            serialized_encrypted_params = serialize_parameters(encrypted_params)
            return [np.frombuffer(serialized_encrypted_params, dtype=np.uint8)]

        def set_parameters(self, parameters):
            serialized_encrypted_params = parameters[0].tobytes()
            encrypted_params = deserialize_parameters(serialized_encrypted_params)
            decrypted_params = decrypt_parameters(encrypted_params, context)
            params_dict = zip(model.state_dict().keys(), decrypted_params)
            state_dict = {k: v.clone().detach() for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            for epoch in tqdm(range(1), desc="Training Epochs"):
                for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader)):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
            return self.get_parameters(), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            _, test_dataset = load_datasets()
            test_loader = DataLoader(test_dataset, batch_size=16)
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc="Evaluating Batches"):
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
