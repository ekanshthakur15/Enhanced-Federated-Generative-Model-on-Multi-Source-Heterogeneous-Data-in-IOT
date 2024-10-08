import torch


class EdgeServer:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_params = []

    def aggregate_params(self):
        """Aggregate parameters from clients."""
        total_params = None
        for i in range(self.num_clients):
            client_params = torch.load(f'client_{i}_params.pt')
            self.client_params.append(client_params)
            if total_params is None:
                total_params = client_params
            else:
                for key in total_params.keys():
                    total_params[key] += client_params[key]
        
        # Average parameters
        for key in total_params.keys():
            total_params[key] /= self.num_clients
        
        return total_params

if __name__ == "__main__":
    edge_server = EdgeServer(num_clients=3)
    aggregated_params = edge_server.aggregate_params()
    torch.save(aggregated_params, 'edge_server_params.pt')  # Save aggregated parameters