import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: Requires `pip install torch-geometric` (Not installed in dev due to Py3.14 limitations)
try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PureSAGEConv(nn.Module):
    """
    Pure-PyTorch implementation of GraphSAGE mean-aggregation convolution.
    Replicates SAGEConv behavior without requiring torch-geometric.
    
    GraphSAGE paper: Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
    
    For each node v:
        h_N(v) = MEAN({h_u : u in N(v)})   # Aggregate neighbor features
        h_v'  = W * CONCAT(h_v, h_N(v))     # Transform self + neighbor
    """
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        # Linear transform: concat(self_features, neighbor_agg) -> out
        self.lin = nn.Linear(in_channels * 2, out_channels)
        self.normalize = normalize
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix (num_nodes, in_channels)
            edge_index: COO edge list (2, num_edges) — row 0 = source, row 1 = target
            edge_weight: Optional edge weights (num_edges,) — unused in mean agg, kept for API compat
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Count incoming edges per target node for mean aggregation
        deg = torch.zeros(num_nodes, dtype=torch.float, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0)  # Avoid division by zero for isolated nodes
        
        # Sum neighbor features into each target node
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        
        # Mean aggregation
        agg = agg / deg.unsqueeze(1)
        
        # Concatenate self-features with neighbor aggregation
        out = torch.cat([x, agg], dim=1)
        out = self.lin(out)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        
        return out


class GraphSAGENetwork(nn.Module):
    """
    Graph Neural Network algorithm mapping the communication topology of the entire LAN.
    Aims to detect Lateral Movement (Scenario 3) by learning normalized interaction graphs.
    
    Uses PyTorch Geometric SAGEConv if available, otherwise falls back to a pure-PyTorch
    implementation of GraphSAGE mean aggregation.
    """
    def __init__(self, num_node_features=14, hidden_channels=32, num_classes=2):
        super(GraphSAGENetwork, self).__init__()
        
        if HAS_PYG:
            # Two GraphSAGE convolution layers for message passing (PyG backend)
            self.conv1 = SAGEConv(num_node_features, hidden_channels, normalize=True)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        else:
            # Pure-PyTorch fallback for environments without torch-geometric
            self.conv1 = PureSAGEConv(num_node_features, hidden_channels, normalize=True)
            self.conv2 = PureSAGEConv(hidden_channels, hidden_channels, normalize=True)
            
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
