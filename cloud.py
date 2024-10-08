import torch


class CloudServer:
    def __init__(self, num_edges):
        self.num_edges = num_edges
        self.edge_params = []

    def aggregate_edge_params(self):
        """Aggregate parameters from edge servers."""
        total_params = None
        for i in range(self.num_edges):
            edge_params = torch.load(f'edge_server_{i}_params.pt')  # Load edge server parameters
            self.edge_params.append(edge_params)
            if total_params is None:
                total_params = edge_params
            else:
                for key in total_params.keys():
                    total_params[key] += edge_params[key]
        
        # Average parameters
        for key in total_params.keys():
            total_params[key] /= self.num_edges
        
        return total_params

if __name__ == "__main__":
    cloud_server = CloudServer(num_edges=3)  # Adjust as necessary
    final_params = cloud_server.aggregate_edge_params()
    torch.save(final_params, 'cloud_gan_params.pt')  # Save final cloud parameters