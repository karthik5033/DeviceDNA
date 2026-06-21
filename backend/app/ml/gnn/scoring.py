"""
GNN Runtime Scorer — Maintains a live communication graph and scores device anomaly
using the pre-trained GraphSAGE model.

Follows the same singleton pattern as LSTMScorer:
- Loads model + normalization params on init
- Gracefully degrades if model files are missing
- Minimum-window guard (5 nodes) before scoring

The runtime graph is built incrementally as the trust engine evaluates devices.
"""
import os
import json
import logging
import torch
import torch.nn.functional as F
import networkx as nx
from app.ml.gnn.model import GraphSAGENetwork

logger = logging.getLogger(__name__)


class GNNScorer:
    def __init__(self):
        self.models_dir = "models_trained/"
        self.model_path = os.path.join(self.models_dir, "gnn_shared.pt")
        self.norm_path = os.path.join(self.models_dir, "gnn_shared_norm.json")
        self.model = None
        self.mins = None
        self.maxs = None
        self.ranges = None
        self.min_nodes = 5  # Minimum graph size before scoring

        # Runtime communication graph — nodes are device_ids, edges are observed communications
        self.graph = nx.DiGraph()
        # Track edge timestamps for dynamic edge expiration (5-minute sliding window)
        self.edge_timestamps = {}
        # Cache latest features per device for building the feature matrix
        self.device_features: dict[str, list[float]] = {}

        self._load_model()

    def _load_model(self):
        """Load pre-trained GNN model and normalization parameters."""
        if os.path.exists(self.model_path):
            try:
                self.model = GraphSAGENetwork(num_node_features=14, hidden_channels=32, num_classes=2)
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
                self.model.eval()
                logger.info("Loaded GNN Shared Model successfully.")
            except Exception as e:
                logger.error(f"Failed to load GNN model: {e}")
                self.model = None
        else:
            logger.warning(f"GNN model not found at {self.model_path}. GNN scoring disabled.")

        if os.path.exists(self.norm_path):
            try:
                with open(self.norm_path, 'r') as f:
                    norm_params = json.load(f)
                self.mins = torch.FloatTensor(norm_params['mins'])
                self.maxs = torch.FloatTensor(norm_params['maxs'])
                self.ranges = self.maxs - self.mins
                self.ranges[self.ranges == 0] = 1.0
                logger.info("Loaded GNN normalization params successfully.")
            except Exception as e:
                logger.error(f"Failed to load GNN norm params: {e}")
        else:
            logger.warning(f"GNN norm params not found at {self.norm_path}. Will use raw features.")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization using stored params."""
        if self.mins is not None:
            return (tensor - self.mins) / self.ranges
        return tensor

    def update_graph(self, src_device_id: str, dst_device_id: str):
        """
        Register an observed communication edge in the runtime graph.
        Called by the trust engine for each evaluation to build the live topology.
        """
        if src_device_id and dst_device_id and src_device_id != dst_device_id:
            import time
            edge = (src_device_id, dst_device_id)
            self.graph.add_edge(src_device_id, dst_device_id)
            self.edge_timestamps[edge] = time.time()

    def prune_expired_edges(self):
        """Prune edges older than 5 minutes (300 seconds) from the graph."""
        import time
        now = time.time()
        expired = [edge for edge, ts in self.edge_timestamps.items() if now - ts > 300]
        for edge in expired:
            src, dst = edge
            if self.graph.has_edge(src, dst):
                self.graph.remove_edge(src, dst)
            if edge in self.edge_timestamps:
                del self.edge_timestamps[edge]

    def update_features(self, device_id: str, features: list[float]):
        """Cache the latest 14D feature vector for a device."""
        self.device_features[device_id] = features
        # Ensure the node exists in the graph even if no edges yet
        if not self.graph.has_node(device_id):
            self.graph.add_node(device_id)

    def score(self, device_id: str, current_features: list[float]) -> float:
        """
        Score a device's anomaly probability using the GraphSAGE model.
        
        Returns softmax P(class=1, anomalous) for the target node, clamped to [0, 1].
        Returns 0.0 if model is unavailable or graph has fewer than min_nodes nodes.
        """
        if self.model is None:
            return 0.0

        # Prune GNN edges older than 5 minutes
        self.prune_expired_edges()

        # Update this device's features in the cache
        self.update_features(device_id, current_features)

        # Minimum window guard — need enough graph context
        if self.graph.number_of_nodes() < self.min_nodes:
            return 0.0

        # Build node ordering: device_id → index
        node_list = list(self.graph.nodes())
        if device_id not in node_list:
            return 0.0

        node_to_idx = {nid: i for i, nid in enumerate(node_list)}
        target_idx = node_to_idx[device_id]

        # Build feature matrix — use cached features, or zeros for nodes with no features yet
        n_nodes = len(node_list)
        feat_matrix = torch.zeros(n_nodes, 14)
        for i, nid in enumerate(node_list):
            if nid in self.device_features:
                feat_matrix[i] = torch.FloatTensor(self.device_features[nid])

        # Normalize features
        feat_matrix = self._normalize(feat_matrix)

        # Build edge_index from networkx graph (bidirectional)
        edges = []
        for src, dst in self.graph.edges():
            if src in node_to_idx and dst in node_to_idx:
                src_idx, dst_idx = node_to_idx[src], node_to_idx[dst]
                edges.append((src_idx, dst_idx))
                edges.append((dst_idx, src_idx))  # Bidirectional

        # Deduplicate
        edges = list(set(edges))

        if not edges:
            # No edges yet — can't do message passing, return 0
            return 0.0

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Forward pass
        with torch.no_grad():
            out = self.model(feat_matrix, edge_index)
            probs = F.softmax(out, dim=1)
            # Return anomaly probability for the target device
            anomaly_score = probs[target_idx, 1].item()
            return max(0.0, min(1.0, anomaly_score))


# Singleton scorer instance
gnn_scorer = GNNScorer()
