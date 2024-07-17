import flwr as fl
import numpy as np

def weighted_average(metrics):
    values = [m["loss"] for m in metrics]
    weights = [m["num_examples"] for m in metrics]
    return np.average(values, weights=weights)

class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Estrazione dei parametri cifrati dai risultati
        parameters_list = [result.parameters.tensors[0] for _, result in results]

        # Aggregazione dei parametri cifrati
        aggregated_parameters = []
        for params in zip(*parameters_list):
            stacked_params = np.stack(params, axis=0)
            aggregated_param = np.mean(stacked_params, axis=0)
            aggregated_parameters.append(aggregated_param)

        aggregated_ndarray = np.array(aggregated_parameters, dtype=np.uint8)
        aggregated_parameters_proto = fl.common.ndarrays_to_parameters([aggregated_ndarray])

        return aggregated_parameters_proto, {}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}

        loss_aggregated = weighted_average([r.metrics for _, r in results])
        return loss_aggregated, {"loss": loss_aggregated}

def main():
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=2,
        min_available_clients=5,
    )

    # Avvia il server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
