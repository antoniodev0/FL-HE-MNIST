from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import logging
from flwr.server.client_proxy import ClientProxy
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
import tenseal as ts
import pickle
from functools import reduce
import numpy as np

from client import cipher, isList, plain

# Load TenSEAL context
public_context = pickle.load(open("public_context.pkl", "rb"))
secret_context = pickle.load(open("secret_context.pkl", "rb"))
context = ts.context_from(secret_context)
context.make_context_public()
public_context = context.serialize()

# Deserialize data to get bytes
def deserializeToBytes(values):
    tempValues = []
    for i in range(len(values)):
        if isinstance(values[i], (list, np.ndarray)):
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
        if isinstance(values[i], (list, np.ndarray)):
            deserialize(values[i], context)
        elif isinstance(values[i], bytes):
            deser = ts.ckks_vector_from(context, values[i])
            tempValues[i] = deser
    return tempValues

def aggregate(results: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    logging.debug("Starting aggregation of weights.")
    num_examples_total = sum([num_examples for _, num_examples in results])
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    
    # Aggiunta di controllo sui tipi di dati e gestione della somma
    aggregated_weights = []
    for layer_updates in zip(*weighted_weights):
        if all(isinstance(layer, np.ndarray) for layer in layer_updates):
            aggregated_layer = sum(layer_updates) / num_examples_total
            aggregated_weights.append(aggregated_layer)
        else:
            logging.error("Layer updates contain non-ndarray types, skipping aggregation.")
            raise TypeError("Layer updates contain non-ndarray types.")
    
    logging.debug("Aggregation complete.")
    return aggregated_weights

# Custom strategy
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        logging.debug(f"Aggregating fit results for round {server_round}.")
        if not results:
            return None, {}

        # Deserialize, decrypt, and ensure data is in np.ndarray format
        weights_results = []
        for _, fit_res in results:
            deserializedBytesParameters = deserializeToBytes(parameters_to_ndarrays(fit_res.parameters))
            deserializedParams = deserialize(deserializedBytesParameters, ts.context_from(public_context))
            private_context = ts.context_from(secret_context)
            decryptedParams = plain(deserializedParams, private_context)
            
            # Forza la conversione in np.ndarray
            ndarray_params = [np.array(param) if not isinstance(param, np.ndarray) else param for param in decryptedParams]
            
            weights_results.append((ndarray_params, fit_res.num_examples))
        
        aggregated_weights = aggregate(weights_results)
        encrypted_aggregated_weights = cipher(aggregated_weights, ts.context_from(public_context))
        return ndarrays_to_parameters(encrypted_aggregated_weights), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]], 
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        logging.debug(f"Aggregating evaluation results for round {server_round}.")
        if not results:
            logging.warning("No results to aggregate. Returning None.")
            return None, {}
        
        total_loss = 0.0
        total_examples = 0
        
        for _, res in results:
            loss = res.loss
            num_examples = res.num_examples
            logging.debug(f"Received loss {loss} with {num_examples} examples.")
            total_loss += loss * num_examples
            total_examples += num_examples
        
        avg_loss = total_loss / total_examples if total_examples > 0 else None
        logging.debug(f"Aggregated average loss: {avg_loss}")
        return avg_loss, {"avg_loss": avg_loss}
def main():
    strategy = CustomFedAvg(
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
