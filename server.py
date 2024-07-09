import flwr as fl
import numpy as np

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if failures:
            return None, {}
        # Aggregate parameters
        aggregated_parameters = []
        for param in zip(*[result.parameters for _, result in results]):
            aggregated_param = np.mean(param, axis=0)
            aggregated_parameters.append(aggregated_param)
        return aggregated_parameters, {}
    
def main():
    strategy = CustomStrategy(
        fraction_fit=0.4,  # 2 di 5 clients (40%) sarà selezionato
        min_fit_clients=2,
        min_available_clients=5,
    )

    # Start Flower server 
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
