import torch
import torch.nn as nn
# Note: Requires `pip install torch-geometric` (Not installed in dev due to Py3.14 limitations)
try:
    from torch_geometric.nn import SAGEConv
    import torch.nn.functional as F
except ImportError:
    # Dummy mock if PyG is not available in bleeding edge envs
    SAGEConv = None
    F = None

class GraphSAGENetwork(nn.Module):
    """
    Graph Neural Network algorithm mapping the communication topology of the entire LAN.
    Aims to detect Lateral Movement (Scenario 3) by learning normalized interaction graphs.
    """
    def __init__(self, num_node_features=14, hidden_channels=32, num_classes=2):
        super(GraphSAGENetwork, self).__init__()
        
        if SAGEConv is None:
            raise ImportError("PyTorch Geometric is required for the GNN Anomaly Pillar.")
            
        # Two GraphSAGE convolution layers for message passing
        self.conv1 = SAGEConv(num_node_features, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        
        # Binary Classification output (0 = Normal, 1 = Edge Anomaly)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Input:
            x: Node feature matrix mapping (num_nodes, 14 dims)
            edge_index: Adjacency list defining network flow paths
            edge_weight: Byte-scale traffic volume on edge
        """
        # First layer of message passing between communication pathways
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second hop passing
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Classification
        out = self.lin(x)
        return out
        
def gnn_anomaly_score(outputs) -> float:
    """Softmaxes node-level logit probabilities into an anomalous probability 0 to 1"""
    probs = F.softmax(outputs, dim=1)
    # The anomaly score is simply the network's confidence in class 1 (Anomalous)
    return probs[:, 1].mean().item()
