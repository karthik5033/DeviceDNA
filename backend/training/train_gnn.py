"""
GNN Training Pipeline — Synthetic IoT Communication Graph + GraphSAGE Training.

Generates a 50-node graph matching the real device fleet distribution from device_profiles.py.
Edges represent realistic IoT communication patterns (cameras → NVR, sensors → MQTT hubs, etc.).
5 nodes are injected as anomalous (lateral movement) with inflated features and cross-VLAN edges.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

# Backend root on sys.path so we can import app.ml modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.gnn.model import GraphSAGENetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models_trained/"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Feature Distribution Profiles (same 14D Gaussians as train_isolation_forest.py) ───
PROFILES = {
    'camera': {
        'means': [80.0, 1200000.0, 1520.0, 782.0, 2500.0, 0.85, 0.15, 0.25, 0.15, 0.10, 0.50, 3.0, 4.0, 0.15],
        'stds':  [15.0, 300000.0,  300.0,  100.0, 500.0,  0.05, 0.05, 0.05, 0.05, 0.02, 0.05, 1.0, 1.0, 0.05]
    },
    'sensor': {
        'means': [20.0, 10000.0, 60.0, 160.0, 2500.0, 0.80, 0.20, 0.20, 0.0, 0.15, 0.65, 2.0, 2.0, 0.01],
        'stds':  [ 5.0,  2000.0, 10.0,  20.0,  500.0, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.0, 0.0, 0.01]
    },
    'thermostat': {
        'means': [15.0, 15000.0, 50.0, 288.0, 2500.0, 0.60, 0.40, 0.0, 0.50, 0.25, 0.25, 3.0, 3.0, 0.30],
        'stds':  [ 5.0,  5000.0, 15.0,  40.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.10]
    },
    'access_control': {
        'means': [30.0, 60000.0, 110.0, 544.0, 2500.0, 0.70, 0.30, 0.0, 0.40, 0.20, 0.40, 4.0, 3.0, 0.10],
        'stds':  [10.0, 20000.0,  30.0,  50.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.05]
    },
    'medical': {
        'means': [40.0, 320000.0, 150.0, 2080.0, 2500.0, 0.80, 0.20, 0.0, 0.30, 0.10, 0.60, 5.0, 4.0, 0.05],
        'stds':  [12.0, 100000.0,  40.0,  200.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 2.0, 1.0, 0.03]
    },
    'industrial': {
        'means': [50.0, 150000.0, 140.0, 1056.0, 2500.0, 0.80, 0.20, 0.0, 0.0, 0.10, 0.90, 3.0, 3.0, 0.02],
        'stds':  [15.0,  40000.0,  35.0,  100.0,  500.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 1.0, 0.0, 0.01]
    }
}

# Device class distribution matching device_profiles.py (total = 50)
CLASS_DISTRIBUTION = [
    ('camera', 12),
    ('sensor', 10),
    ('thermostat', 8),
    ('access_control', 6),
    ('medical', 8),
    ('industrial', 6),
]

# VLAN assignments per device class (matches device_profiles.py)
VLAN_MAP = {
    'camera': 20,
    'sensor': 10,
    'thermostat': 10,
    'access_control': 20,
    'medical': 30,
    'industrial': 40,
}


def generate_node_features(rng: np.random.Generator) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Generate 14D feature vectors for 50 nodes using class-specific Gaussian distributions.
    Returns: (features [50, 14], class_labels [50], vlan_ids [50])
    """
    features = []
    class_labels = []
    vlan_ids = []

    for cls_name, count in CLASS_DISTRIBUTION:
        means = np.array(PROFILES[cls_name]['means'])
        stds = np.array(PROFILES[cls_name]['stds'])

        for _ in range(count):
            feat = rng.normal(loc=means, scale=stds)
            feat = np.maximum(feat, 0.0)  # Clamp non-negative
            features.append(feat)
            class_labels.append(cls_name)
            vlan_ids.append(VLAN_MAP[cls_name])

    return np.array(features, dtype=np.float32), class_labels, vlan_ids


def build_topology_edges(class_labels: list[str], vlan_ids: list[int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """
    Build edges representing realistic IoT communication patterns:
    - Intra-VLAN mesh: devices in the same VLAN connect to each other (sparse)
    - Gateway links: each device connects to a virtual gateway node at index 0 of its VLAN group
    - Cross-VLAN sparse bridges: gateways connect across VLANs
    """
    edges = []
    n = len(class_labels)

    # Group nodes by VLAN
    vlan_groups: dict[int, list[int]] = {}
    for i, vlan in enumerate(vlan_ids):
        vlan_groups.setdefault(vlan, []).append(i)

    # 1. Intra-VLAN edges: each node connects to 1-3 peers within its VLAN
    for vlan, members in vlan_groups.items():
        for node in members:
            n_peers = min(rng.integers(1, 4), len(members) - 1)
            peers = [m for m in members if m != node]
            if peers:
                chosen = rng.choice(peers, size=min(n_peers, len(peers)), replace=False)
                for peer in chosen:
                    edges.append((node, peer))

    # 2. Gateway hub links: first node in each VLAN acts as gateway/hub
    gateway_nodes = []
    for vlan, members in vlan_groups.items():
        gateway = members[0]
        gateway_nodes.append(gateway)
        for node in members[1:]:
            edges.append((node, gateway))

    # 3. Cross-VLAN sparse bridges between gateways
    for i, gw_a in enumerate(gateway_nodes):
        for gw_b in gateway_nodes[i + 1:]:
            if rng.random() < 0.6:  # 60% chance of inter-VLAN gateway link
                edges.append((gw_a, gw_b))

    # Deduplicate
    edge_set = set()
    for src, dst in edges:
        if src != dst:
            edge_set.add((src, dst))

    return list(edge_set)


def inject_anomalies(
    features: np.ndarray,
    edges: list[tuple[int, int]],
    class_labels: list[str],
    vlan_ids: list[int],
    rng: np.random.Generator,
    n_anomalous: int = 5
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """
    Select n_anomalous random nodes and inject lateral movement signatures:
    - Dramatically inflated unique_dst_ips (feature[11]) by 10-15×
    - Spiked external_traffic_ratio (feature[13]) to 0.6-0.9
    - Elevated total_bytes (feature[1]) by 3-5×
    - Cross-VLAN edges violating normal topology
    
    Returns: (modified_features, modified_edges, labels [0=normal, 1=anomalous])
    """
    n = features.shape[0]
    labels = np.zeros(n, dtype=np.int64)

    anomalous_nodes = rng.choice(n, size=n_anomalous, replace=False)
    labels[anomalous_nodes] = 1

    for node in anomalous_nodes:
        # Inflate unique destination IPs (feature index 11)
        features[node, 11] *= rng.uniform(10.0, 15.0)

        # Spike external traffic ratio (feature index 13)
        features[node, 13] = rng.uniform(0.6, 0.9)

        # Elevate total bytes (feature index 1)
        features[node, 1] *= rng.uniform(3.0, 5.0)

        # Add cross-VLAN edges (lateral movement)
        node_vlan = vlan_ids[node]
        cross_vlan_targets = [i for i in range(n) if vlan_ids[i] != node_vlan]
        if cross_vlan_targets:
            n_cross = rng.integers(2, 5)
            targets = rng.choice(cross_vlan_targets, size=min(n_cross, len(cross_vlan_targets)), replace=False)
            for t in targets:
                edges.append((node, int(t)))

    return features, edges, labels


def to_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Convert directed edge list to bidirectional COO edge_index tensor (2, 2*E)."""
    bidir = []
    for src, dst in edges:
        bidir.append((src, dst))
        bidir.append((dst, src))

    # Deduplicate
    bidir = list(set(bidir))

    if not bidir:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(bidir, dtype=torch.long).t().contiguous()


def train():
    rng = np.random.default_rng(seed=42)

    # ─── 1. Build synthetic graph ───
    logger.info("Generating synthetic IoT communication graph (50 nodes)...")

    features, class_labels, vlan_ids = generate_node_features(rng)
    edges = build_topology_edges(class_labels, vlan_ids, rng)

    logger.info(f"  Normal graph: {features.shape[0]} nodes, {len(edges)} directed edges")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(class_labels, return_counts=True)))}")

    # ─── 2. Inject anomalies ───
    features, edges, labels = inject_anomalies(features, edges, class_labels, vlan_ids, rng, n_anomalous=5)
    anomalous_indices = np.where(labels == 1)[0]
    logger.info(f"  Injected anomalies at nodes: {anomalous_indices.tolist()}")
    logger.info(f"  Post-injection edges: {len(edges)}")

    # ─── 3. Normalize features ───
    feat_tensor = torch.FloatTensor(features)
    mins = feat_tensor.min(dim=0).values
    maxs = feat_tensor.max(dim=0).values
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    feat_norm = (feat_tensor - mins) / ranges

    # ─── 4. Build PyTorch tensors ───
    edge_index = to_edge_index(edges)
    labels_tensor = torch.LongTensor(labels)

    logger.info(f"  Feature matrix: {feat_norm.shape}")
    logger.info(f"  Edge index: {edge_index.shape}")
    logger.info(f"  Labels: {labels_tensor.shape} (anomalous={labels.sum()}, normal={len(labels) - labels.sum()})")

    # ─── 5. Train GraphSAGE ───
    model = GraphSAGENetwork(num_node_features=14, hidden_channels=32, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(
        # Class imbalance weighting: 45 normal vs 5 anomalous → weight anomalous 9× more
        weight=torch.FloatTensor([1.0, 9.0])
    )

    epochs = 30
    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(feat_norm, edge_index)
        loss = criterion(out, labels_tensor)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = out.argmax(dim=1)
            acc = (preds == labels_tensor).float().mean().item()
            anom_correct = (preds[labels_tensor == 1] == 1).float().mean().item() if labels.sum() > 0 else 0.0

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:02d}/{epochs} — Loss: {loss.item():.4f} | Acc: {acc:.3f} | Anomaly Recall: {anom_correct:.3f}")

    # ─── 6. Save model + normalization params ───
    model_path = os.path.join(MODELS_DIR, "gnn_shared.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ GNN Model saved to {model_path}")

    norm_path = os.path.join(MODELS_DIR, "gnn_shared_norm.json")
    with open(norm_path, 'w') as f:
        json.dump({
            'mins': mins.tolist(),
            'maxs': maxs.tolist()
        }, f)
    logger.info(f"✅ GNN Norm Params saved to {norm_path}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_out = model(feat_norm, edge_index)
        final_preds = final_out.argmax(dim=1)
        final_acc = (final_preds == labels_tensor).float().mean().item()
        logger.info(f"✅ Final Accuracy: {final_acc:.3f}")
        logger.info(f"   Predictions: {final_preds.tolist()}")
        logger.info(f"   Ground Truth: {labels_tensor.tolist()}")


if __name__ == "__main__":
    train()
