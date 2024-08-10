import flwr as fl
from flwr.server.strategy import FedAvg

import tenseal as ts
import numpy as np
import pickle

# Carica il contesto TenSEAL
with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

context = ts.context_from(public_context)

class HomomorphicFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model parameters using homomorphic encryption."""
        if not results:
            return None, {}

        # Deserializza i parametri cifrati in CKKSVector e collega il contesto
        encrypted_params = []
        for _, fit_res in results:
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)  # Collegare il contesto dopo la deserializzazione
                client_params.append(ckks_vector)
            encrypted_params.append(client_params)

        # Somma i parametri cifrati di tutti i client
        aggregated_params = encrypted_params[0]
        for client_params in encrypted_params[1:]:
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i]

        # Dividi per il numero di client moltiplicando per l'inverso
        num_clients = len(encrypted_params)
        inverse_num_clients = 1.0 / num_clients  # Calcola l'inverso
        for i in range(len(aggregated_params)):
            aggregated_params[i] = aggregated_params[i] * inverse_num_clients  # Moltiplica per l'inverso

        # Serializza i parametri aggregati e ritorna come Parameters
        aggregated_serialized = [param.serialize() for param in aggregated_params]
        return fl.common.ndarrays_to_parameters(aggregated_serialized), {}

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggrega la perdita (val_loss) in chiaro dei client
        total_loss = sum(r.metrics["val_loss"] * r.num_examples for _, r in results)
        total_samples = sum(r.num_examples for _, r in results)
        average_loss = total_loss / total_samples

        return average_loss, {}

def main():
    strategy = HomomorphicFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
