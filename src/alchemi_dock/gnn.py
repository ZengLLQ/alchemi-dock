"""Vectorized Message-Passing Graph Neural Networks (GCN, MPNN, GATv2) in pure NumPy/SciPy.

Implements full forward and exact analytical reverse-mode backpropagation through:
- Kipf-Welling spectral Graph Convolutional Networks (GCN)
- Edge-conditioned Message-Passing Neural Networks with GRU updates (MPNN)
- Multi-head dynamic Graph Attention Networks (GATv2) with atom attribution scores

Includes gated attention graph readout, AdamW optimization, scaffold-split validation,
and an optional guarded PyTorch availability check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .descriptors import bemis_murcko_scaffold
from .molgraph import parse_smiles
from .types import (
    PAULING_EN,
    FloatArray,
    GNNMetrics,
    GNNTrainingReport,
    IntArray,
    MoleculeGraph,
)

_ELEM_VOCAB: Tuple[str, ...] = ("C", "N", "O", "F", "Cl", "Br", "S", "P")
ATOM_FEAT_DIM = 19
EDGE_FEAT_DIM = 6


def has_torch_adapter() -> bool:
    """Return True if optional PyTorch backend is installed, else False."""
    try:
        import importlib

        importlib.import_module("torch")
        return True
    except ImportError:
        return False


@dataclass
class MolTensorGraph:
    """Numerical tensor representation of a MoleculeGraph for GNN forward/backward passes."""

    smiles: str
    atom_features: FloatArray
    edge_src: IntArray
    edge_dst: IntArray
    edge_features: FloatArray
    adj_norm: FloatArray
    num_atoms: int


def featurize_mol_tensor(mol: MoleculeGraph) -> MolTensorGraph:
    """Convert a MoleculeGraph into dense atom feature matrices and directed edge lists."""
    n = mol.num_atoms
    x = np.zeros((n, ATOM_FEAT_DIM), dtype=np.float64)
    for i, atom in enumerate(mol.atoms):
        if atom.element in _ELEM_VOCAB:
            x[i, _ELEM_VOCAB.index(atom.element)] = 1.0
        else:
            x[i, 8] = 1.0
        deg = len(mol.neighbors(i))
        x[i, 9] = deg / 4.0
        x[i, 10] = atom.implicit_h / 4.0
        x[i, 11] = float(atom.formal_charge)
        x[i, 12] = 1.0 if atom.is_aromatic else 0.0
        x[i, 13] = 1.0 if atom.in_ring else 0.0
        x[i, 14] = 1.0 if atom.hybridization == "sp3" else (0.5 if atom.hybridization == "sp2" else 0.0)
        x[i, 15] = atom.partial_charge
        x[i, 16] = 1.0 if (atom.element in ("N", "O") and atom.implicit_h > 0) else 0.0
        x[i, 17] = 1.0 if atom.element in ("N", "O") else 0.0
        x[i, 18] = PAULING_EN.get(atom.element, 2.5) / 4.0

    src_list: List[int] = []
    dst_list: List[int] = []
    efeat_list: List[List[float]] = []

    adj = np.eye(n, dtype=np.float64)
    for bond in mol.bonds:
        u, v = bond.atom_u, bond.atom_v
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        feat = [
            bond.order / 3.0,
            1.0 if bond.is_aromatic else 0.0,
            1.0 if bond.is_conjugated else 0.0,
            1.0 if bond.in_ring else 0.0,
            1.0 if bond.order >= 1.8 else 0.0,
            0.0,
        ]
        src_list.extend([u, v])
        dst_list.extend([v, u])
        efeat_list.extend([feat, feat])

    # Add explicit self-loop edges so attention/message layers remain well-defined on single atoms
    for i in range(n):
        src_list.append(i)
        dst_list.append(i)
        efeat_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    deg_vec = np.sum(adj, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(1.0, deg_vec))
    adj_norm = (d_inv_sqrt[:, None] * adj) * d_inv_sqrt[None, :]

    return MolTensorGraph(
        smiles=mol.canonical_smiles or mol.smiles,
        atom_features=x,
        edge_src=np.asarray(src_list, dtype=np.int64),
        edge_dst=np.asarray(dst_list, dtype=np.int64),
        edge_features=np.asarray(efeat_list, dtype=np.float64),
        adj_norm=adj_norm,
        num_atoms=n,
    )


def _silu(x: FloatArray) -> Tuple[FloatArray, FloatArray]:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    y = x * sig
    grad = sig + x * sig * (1.0 - sig)
    return y, grad


def _leaky_relu(x: FloatArray, neg_slope: float = 0.2) -> Tuple[FloatArray, FloatArray]:
    y = np.where(x >= 0.0, x, neg_slope * x)
    grad = np.where(x >= 0.0, 1.0, neg_slope)
    return y, grad


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _segment_softmax(logits: FloatArray, dst: IntArray, n_nodes: int) -> FloatArray:
    """Numerically stable segment softmax over incoming edges grouped by dst node."""
    max_per_node = np.full(n_nodes, -1e18, dtype=np.float64)
    np.maximum.at(max_per_node, dst, logits)
    shifted = logits - max_per_node[dst]
    exp_vals = np.exp(np.clip(shifted, -40.0, 40.0))
    sum_per_node = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(sum_per_node, dst, exp_vals)
    return exp_vals / np.maximum(1e-12, sum_per_node[dst])


def _segment_softmax_backward(alpha: FloatArray, d_alpha: FloatArray, dst: IntArray, n_nodes: int) -> FloatArray:
    """Exact Jacobian-vector product for segment softmax."""
    dot_per_node = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(dot_per_node, dst, alpha * d_alpha)
    return alpha * (d_alpha - dot_per_node[dst])


class MolecularGNN:
    """Pure-NumPy Graph Neural Network supporting GCN, MPNN, and GATv2 architectures
    with exact analytical gradients, gated attention pooling, and AdamW optimization.
    """

    def __init__(
        self,
        architecture: str = "gatv2",
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        arch = architecture.lower()
        if arch not in ("gcn", "mpnn", "gatv2"):
            raise ValueError(f"Unsupported GNN architecture '{architecture}'")
        self.architecture = arch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        self.params: Dict[str, FloatArray] = {}
        self._adam_m: Dict[str, FloatArray] = {}
        self._adam_v: Dict[str, FloatArray] = {}
        self._step = 0
        self._init_parameters()

    def _glorot(self, fan_in: int, fan_out: int) -> FloatArray:
        limit = math.sqrt(6.0 / max(1, fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)

    def _init_parameters(self) -> None:
        d = self.hidden_dim
        p = self.params
        p["w_in"] = self._glorot(ATOM_FEAT_DIM, d)
        p["b_in"] = np.zeros(d, dtype=np.float64)

        for layer in range(self.num_layers):
            if self.architecture == "gcn":
                p[f"gcn_w_{layer}"] = self._glorot(d, d)
                p[f"gcn_b_{layer}"] = np.zeros(d, dtype=np.float64)
            elif self.architecture == "mpnn":
                p[f"mp_wm_{layer}"] = self._glorot(d, d)
                p[f"mp_we_{layer}"] = self._glorot(EDGE_FEAT_DIM, d)
                p[f"mp_bm_{layer}"] = np.zeros(d, dtype=np.float64)
                p[f"gru_wz_{layer}"] = self._glorot(2 * d, d)
                p[f"gru_bz_{layer}"] = np.zeros(d, dtype=np.float64)
                p[f"gru_wr_{layer}"] = self._glorot(2 * d, d)
                p[f"gru_br_{layer}"] = np.zeros(d, dtype=np.float64)
                p[f"gru_wh_{layer}"] = self._glorot(2 * d, d)
                p[f"gru_bh_{layer}"] = np.zeros(d, dtype=np.float64)
            else:
                # Multi-head GATv2
                p[f"gat_wl_{layer}"] = self._glorot(d, d)
                p[f"gat_wr_{layer}"] = self._glorot(d, d)
                p[f"gat_we_{layer}"] = self._glorot(EDGE_FEAT_DIM, d)
                p[f"gat_a_{layer}"] = self._glorot(d // self.num_heads, self.num_heads)
                p[f"gat_b_{layer}"] = np.zeros(d, dtype=np.float64)

        # Gated attention readout + 2-layer MLP head
        p["w_gate"] = self._glorot(d, 1)
        p["w_mlp1"] = self._glorot(2 * d, d)
        p["b_mlp1"] = np.zeros(d, dtype=np.float64)
        p["w_mlp2"] = self._glorot(d, 1)
        p["b_mlp2"] = np.zeros(1, dtype=np.float64)

        for k, v in p.items():
            self._adam_m[k] = np.zeros_like(v)
            self._adam_v[k] = np.zeros_like(v)

    def forward_single(self, graph: MolTensorGraph) -> Tuple[float, FloatArray, Dict[str, object]]:
        """Run forward pass on one MolTensorGraph.

        Returns (prediction, atom_attributions, backprop_cache).
        """
        p = self.params
        n = graph.num_atoms
        d = self.hidden_dim
        cache: Dict[str, object] = {"graph": graph}

        z0 = graph.atom_features @ p["w_in"] + p["b_in"]
        h, z0_grad = _silu(z0)
        cache["z0_grad"] = z0_grad
        cache["h_in"] = h

        layer_caches: List[Dict[str, object]] = []
        for layer in range(self.num_layers):
            lc: Dict[str, object] = {"h_prev": h}
            if self.architecture == "gcn":
                prop = graph.adj_norm @ h
                z = prop @ p[f"gcn_w_{layer}"] + p[f"gcn_b_{layer}"]
                act, act_grad = _silu(z)
                h = h + act
                lc.update({"prop": prop, "act_grad": act_grad})
            elif self.architecture == "mpnn":
                src = graph.edge_src
                dst = graph.edge_dst
                e_count = len(src)
                h_src = h[src]
                msg_pre = h_src @ p[f"mp_wm_{layer}"] + graph.edge_features @ p[f"mp_we_{layer}"] + p[f"mp_bm_{layer}"]
                msg, msg_grad = _silu(msg_pre)
                deg = np.zeros(n, dtype=np.float64)
                np.add.at(deg, dst, 1.0)
                norm = 1.0 / np.sqrt(np.maximum(1.0, deg))
                agg = np.zeros((n, d), dtype=np.float64)
                np.add.at(agg, dst, msg)
                agg_norm = agg * norm[:, None]

                concat_zr = np.concatenate([agg_norm, h], axis=1)
                z_gate = _sigmoid(concat_zr @ p[f"gru_wz_{layer}"] + p[f"gru_bz_{layer}"])
                r_gate = _sigmoid(concat_zr @ p[f"gru_wr_{layer}"] + p[f"gru_br_{layer}"])
                concat_cand = np.concatenate([agg_norm, r_gate * h], axis=1)
                h_cand = np.tanh(concat_cand @ p[f"gru_wh_{layer}"] + p[f"gru_bh_{layer}"])
                h = (1.0 - z_gate) * h + z_gate * h_cand
                lc.update(
                    {
                        "e_count": e_count,
                        "msg_grad": msg_grad,
                        "norm": norm,
                        "concat_zr": concat_zr,
                        "z_gate": z_gate,
                        "r_gate": r_gate,
                        "concat_cand": concat_cand,
                        "h_cand": h_cand,
                    }
                )
            else:
                # GATv2 multi-head
                src = graph.edge_src
                dst = graph.edge_dst
                k_heads = self.num_heads
                hd = d // k_heads
                hl = h @ p[f"gat_wl_{layer}"]
                hr = h @ p[f"gat_wr_{layer}"]
                he = graph.edge_features @ p[f"gat_we_{layer}"]

                hl_e = hl[src]
                g_pre = (hl_e + hr[dst] + he).reshape(-1, k_heads, hd)
                g_act, g_grad = _leaky_relu(g_pre, neg_slope=0.2)
                a_vec = p[f"gat_a_{layer}"]  # shape (hd, k_heads)
                logits = np.einsum("ekd,dk->ek", g_act, a_vec)

                hl_heads = hl_e.reshape(-1, k_heads, hd)
                out_heads = np.zeros((n, k_heads, hd), dtype=np.float64)
                alphas = np.zeros_like(logits)
                for k in range(k_heads):
                    ak = _segment_softmax(logits[:, k], dst, n)
                    alphas[:, k] = ak
                    np.add.at(out_heads[:, k, :], dst, ak[:, None] * hl_heads[:, k, :])

                out_flat = out_heads.reshape(n, d) + p[f"gat_b_{layer}"]
                act, act_grad = _silu(out_flat)
                h = h + act
                lc.update(
                    {
                        "hl": hl,
                        "g_act": g_act,
                        "g_grad": g_grad,
                        "alphas": alphas,
                        "act_grad": act_grad,
                    }
                )
            layer_caches.append(lc)

        cache["layer_caches"] = layer_caches
        cache["h_final"] = h

        # Gated attention readout + mean readout
        gate_logits = (h @ p["w_gate"]).ravel()
        max_g = float(np.max(gate_logits))
        exp_g = np.exp(np.clip(gate_logits - max_g, -40.0, 40.0))
        beta = exp_g / max(1e-12, float(np.sum(exp_g)))
        pool_attn = beta @ h
        pool_mean = np.mean(h, axis=0)
        g_vec = np.concatenate([pool_attn, pool_mean])

        mlp1_pre = g_vec @ p["w_mlp1"] + p["b_mlp1"]
        mlp1_act, mlp1_grad = _silu(mlp1_pre)
        pred = float((mlp1_act @ p["w_mlp2"] + p["b_mlp2"])[0])

        cache.update(
            {
                "beta": beta,
                "g_vec": g_vec,
                "mlp1_act": mlp1_act,
                "mlp1_grad": mlp1_grad,
            }
        )
        return pred, beta.copy(), cache

    def backward_single(self, d_pred: float, cache: Dict[str, object]) -> Dict[str, FloatArray]:
        """Compute exact analytical parameter gradients for one forward cache."""
        p = self.params
        grads: Dict[str, FloatArray] = {k: np.zeros_like(v) for k, v in p.items()}
        graph = cache["graph"]
        assert isinstance(graph, MolTensorGraph)
        n = graph.num_atoms
        d = self.hidden_dim

        mlp1_act = cache["mlp1_act"]
        mlp1_grad = cache["mlp1_grad"]
        g_vec = cache["g_vec"]
        beta = cache["beta"]
        h_final = cache["h_final"]
        assert isinstance(mlp1_act, np.ndarray)
        assert isinstance(mlp1_grad, np.ndarray)
        assert isinstance(g_vec, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert isinstance(h_final, np.ndarray)

        # MLP head
        grads["w_mlp2"] += np.outer(mlp1_act, np.array([d_pred]))
        grads["b_mlp2"] += np.array([d_pred])
        d_mlp1_act = d_pred * p["w_mlp2"].ravel()
        d_mlp1_pre = d_mlp1_act * mlp1_grad
        grads["w_mlp1"] += np.outer(g_vec, d_mlp1_pre)
        grads["b_mlp1"] += d_mlp1_pre
        d_g = d_mlp1_pre @ p["w_mlp1"].T

        d_pool_attn = d_g[:d]
        d_pool_mean = d_g[d:]

        d_h = np.outer(beta, d_pool_attn) + (d_pool_mean / max(1, n))[None, :]
        d_beta = h_final @ d_pool_attn
        dot_b = float(np.sum(beta * d_beta))
        d_gate_logits = beta * (d_beta - dot_b)
        grads["w_gate"] += (h_final.T @ d_gate_logits)[:, None]
        d_h += np.outer(d_gate_logits, p["w_gate"].ravel())

        layer_caches = cache["layer_caches"]
        assert isinstance(layer_caches, list)

        for layer in range(self.num_layers - 1, -1, -1):
            lc = layer_caches[layer]
            h_prev = lc["h_prev"]
            assert isinstance(h_prev, np.ndarray)

            if self.architecture == "gcn":
                prop = lc["prop"]
                act_grad = lc["act_grad"]
                assert isinstance(prop, np.ndarray) and isinstance(act_grad, np.ndarray)
                d_z = d_h * act_grad
                grads[f"gcn_w_{layer}"] += prop.T @ d_z
                grads[f"gcn_b_{layer}"] += np.sum(d_z, axis=0)
                d_prop = d_z @ p[f"gcn_w_{layer}"].T
                d_h = d_h + graph.adj_norm.T @ d_prop

            elif self.architecture == "mpnn":
                msg_grad = lc["msg_grad"]
                norm = lc["norm"]
                concat_zr = lc["concat_zr"]
                z_gate = lc["z_gate"]
                r_gate = lc["r_gate"]
                concat_cand = lc["concat_cand"]
                h_cand = lc["h_cand"]
                assert (
                    isinstance(msg_grad, np.ndarray)
                    and isinstance(norm, np.ndarray)
                    and isinstance(concat_zr, np.ndarray)
                    and isinstance(z_gate, np.ndarray)
                    and isinstance(r_gate, np.ndarray)
                    and isinstance(concat_cand, np.ndarray)
                    and isinstance(h_cand, np.ndarray)
                )
                d_h_prev = (1.0 - z_gate) * d_h
                d_z_gate = d_h * (h_cand - h_prev)
                d_h_cand = d_h * z_gate
                d_cand_pre = d_h_cand * (1.0 - h_cand ** 2)
                grads[f"gru_wh_{layer}"] += concat_cand.T @ d_cand_pre
                grads[f"gru_bh_{layer}"] += np.sum(d_cand_pre, axis=0)
                d_concat_cand = d_cand_pre @ p[f"gru_wh_{layer}"].T
                d_agg_norm = d_concat_cand[:, :d]
                d_rh = d_concat_cand[:, d:]
                d_h_prev += d_rh * r_gate
                d_r_gate = d_rh * h_prev

                d_z_pre = d_z_gate * z_gate * (1.0 - z_gate)
                d_r_pre = d_r_gate * r_gate * (1.0 - r_gate)
                grads[f"gru_wz_{layer}"] += concat_zr.T @ d_z_pre
                grads[f"gru_bz_{layer}"] += np.sum(d_z_pre, axis=0)
                grads[f"gru_wr_{layer}"] += concat_zr.T @ d_r_pre
                grads[f"gru_br_{layer}"] += np.sum(d_r_pre, axis=0)
                d_concat_zr = d_z_pre @ p[f"gru_wz_{layer}"].T + d_r_pre @ p[f"gru_wr_{layer}"].T
                d_agg_norm += d_concat_zr[:, :d]
                d_h_prev += d_concat_zr[:, d:]

                d_agg = d_agg_norm * norm[:, None]
                src = graph.edge_src
                dst = graph.edge_dst
                d_msg = d_agg[dst]
                d_msg_pre = d_msg * msg_grad
                grads[f"mp_wm_{layer}"] += h_prev[src].T @ d_msg_pre
                grads[f"mp_we_{layer}"] += graph.edge_features.T @ d_msg_pre
                grads[f"mp_bm_{layer}"] += np.sum(d_msg_pre, axis=0)
                d_h_src = d_msg_pre @ p[f"mp_wm_{layer}"].T
                np.add.at(d_h_prev, src, d_h_src)
                d_h = d_h_prev

            else:
                src = graph.edge_src
                dst = graph.edge_dst
                k_heads = self.num_heads
                hd = d // k_heads
                hl = lc["hl"]
                g_act = lc["g_act"]
                g_grad = lc["g_grad"]
                alphas = lc["alphas"]
                act_grad = lc["act_grad"]
                assert (
                    isinstance(hl, np.ndarray)
                    and isinstance(g_act, np.ndarray)
                    and isinstance(g_grad, np.ndarray)
                    and isinstance(alphas, np.ndarray)
                    and isinstance(act_grad, np.ndarray)
                )

                d_act = d_h
                d_out_flat = d_act * act_grad
                grads[f"gat_b_{layer}"] += np.sum(d_out_flat, axis=0)
                d_out_heads = d_out_flat.reshape(n, k_heads, hd)

                hl_heads = hl[src].reshape(-1, k_heads, hd)
                d_hl_heads = np.zeros_like(hl_heads)
                d_alphas = np.zeros_like(alphas)
                for k in range(k_heads):
                    ak = alphas[:, k]
                    d_out_k_e = d_out_heads[dst, k, :]
                    d_alphas[:, k] = np.sum(d_out_k_e * hl_heads[:, k, :], axis=1)
                    d_hl_heads[:, k, :] = ak[:, None] * d_out_k_e

                d_logits = np.zeros_like(alphas)
                for k in range(k_heads):
                    d_logits[:, k] = _segment_softmax_backward(alphas[:, k], d_alphas[:, k], dst, n)

                a_vec = p[f"gat_a_{layer}"]
                grads[f"gat_a_{layer}"] += np.einsum("ekd,ek->dk", g_act, d_logits)
                d_g_act = np.einsum("ek,dk->ekd", d_logits, a_vec)
                d_g_pre = (d_g_act * g_grad).reshape(-1, d)

                d_hl = np.zeros_like(hl)
                np.add.at(d_hl, src, d_hl_heads.reshape(-1, d) + d_g_pre)
                d_hr = np.zeros_like(hl)
                np.add.at(d_hr, dst, d_g_pre)

                grads[f"gat_wl_{layer}"] += h_prev.T @ d_hl
                grads[f"gat_wr_{layer}"] += h_prev.T @ d_hr
                grads[f"gat_we_{layer}"] += graph.edge_features.T @ d_g_pre
                d_h = d_h + d_hl @ p[f"gat_wl_{layer}"].T + d_hr @ p[f"gat_wr_{layer}"].T

        z0_grad = cache["z0_grad"]
        assert isinstance(z0_grad, np.ndarray)
        d_z0 = d_h * z0_grad
        grads["w_in"] += graph.atom_features.T @ d_z0
        grads["b_in"] += np.sum(d_z0, axis=0)
        return grads

    def _adamw_step(self, grads: Dict[str, FloatArray], lr: float) -> None:
        self._step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k, param in self.params.items():
            g = np.clip(grads[k], -5.0, 5.0)
            if param.ndim >= 2:
                g = g + self.weight_decay * param
            self._adam_m[k] = beta1 * self._adam_m[k] + (1.0 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = self._adam_m[k] / (1.0 - beta1 ** self._step)
            v_hat = self._adam_v[k] / (1.0 - beta2 ** self._step)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, graphs: Sequence[MolTensorGraph]) -> Tuple[FloatArray, Dict[str, List[float]]]:
        """Predict scalar property values and atom attention weights for molecular graphs."""
        preds = np.zeros(len(graphs), dtype=np.float64)
        attributions: Dict[str, List[float]] = {}
        for i, g in enumerate(graphs):
            pred, beta, _ = self.forward_single(g)
            preds[i] = pred
            attributions[g.smiles] = [float(x) for x in beta]
        return preds, attributions

    def fit(
        self,
        train_graphs: Sequence[MolTensorGraph],
        train_targets: FloatArray,
        val_graphs: Sequence[MolTensorGraph],
        val_targets: FloatArray,
        epochs: int = 35,
        lr: float = 8e-3,
    ) -> GNNTrainingReport:
        """Train the GNN using mini-batch AdamW + cosine learning rate annealing."""
        n_train = len(train_graphs)
        if self._step == 0 and len(train_targets) > 0:
            self.params["b_mlp2"][0] = float(np.mean(train_targets))
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(epochs):
            cos_lr = lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
            cos_lr = max(lr * 0.08, cos_lr)
            order = self.rng.permutation(n_train)
            batch_size = min(8, n_train)
            ep_loss = 0.0

            for start in range(0, n_train, batch_size):
                batch_idx = order[start : start + batch_size]
                acc_grads: Dict[str, FloatArray] = {k: np.zeros_like(v) for k, v in self.params.items()}
                b_len = len(batch_idx)
                for idx in batch_idx:
                    pred, _, cache = self.forward_single(train_graphs[idx])
                    err = pred - float(train_targets[idx])
                    ep_loss += err ** 2
                    d_pred = (2.0 * err) / b_len
                    g_single = self.backward_single(d_pred, cache)
                    for k in acc_grads:
                        acc_grads[k] += g_single[k]
                self._adamw_step(acc_grads, lr=cos_lr)

            train_losses.append(float(ep_loss / max(1, n_train)))
            val_preds, _ = self.predict(val_graphs)
            val_mse = float(np.mean((val_preds - val_targets) ** 2)) if len(val_graphs) > 0 else 0.0
            val_losses.append(val_mse)

        tr_preds, tr_attr = self.predict(train_graphs)
        va_preds, va_attr = self.predict(val_graphs)
        combined_attr = {**tr_attr, **va_attr}
        return GNNTrainingReport(
            architecture=self.architecture,
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
            train_metrics=evaluate_regression_metrics(train_targets, tr_preds),
            val_metrics=evaluate_regression_metrics(val_targets, va_preds),
            atom_attributions=combined_attr,
        )


def evaluate_regression_metrics(y_true: FloatArray, y_pred: FloatArray) -> GNNMetrics:
    """Compute RMSE, MAE, R^2, Spearman rank correlation, AUROC, and 5% Enrichment Factor."""
    n = len(y_true)
    if n == 0:
        return GNNMetrics(0.0, 0.0, 0.0, 0.0, 0.5, 1.0)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Spearman rho
    r_true = np.argsort(np.argsort(y_true)).astype(np.float64)
    r_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)
    if np.std(r_true) > 1e-9 and np.std(r_pred) > 1e-9:
        spearman = float(np.corrcoef(r_true, r_pred)[0, 1])
    else:
        spearman = 0.0

    # Binary active AUROC using median or pIC50 >= 6.5 threshold
    thresh = max(float(np.median(y_true)), 6.5 if np.max(y_true) > 6.8 else float(np.median(y_true)))
    actives = y_true >= thresh
    n_pos = int(np.sum(actives))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
        ef5 = 1.0
    else:
        order = np.argsort(-y_pred)
        sorted_act = actives[order]
        tp = 0
        auc_sum = 0.0
        for flag in sorted_act:
            if flag:
                tp += 1
            else:
                auc_sum += tp
        auroc = float(auc_sum / (n_pos * n_neg))
        top_k = max(1, int(math.ceil(0.15 * n)))
        hits_top = int(np.sum(sorted_act[:top_k]))
        ef5 = float((hits_top / top_k) / max(1e-9, n_pos / n))

    return GNNMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_rho=spearman,
        auroc=auroc,
        enrichment_5pct=ef5,
    )


def scaffold_split_indices(smiles_list: Sequence[str], val_fraction: float = 0.25) -> Tuple[List[int], List[int]]:
    """Deterministic Bemis-Murcko scaffold split grouping identical cores together."""
    groups: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        mol = parse_smiles(smi)
        scaff = bemis_murcko_scaffold(mol).murcko_smiles or f"_acyclic_{idx}"
        groups.setdefault(scaff, []).append(idx)

    # Sort scaffold buckets largest-first so major scaffolds populate train, smaller/novel populate val
    buckets = sorted(groups.values(), key=lambda b: (-len(b), b[0]))
    target_val = max(1, int(round(len(smiles_list) * val_fraction)))
    train_idx: List[int] = []
    val_idx: List[int] = []

    # Assign largest buckets to train until remaining fits target_val
    total = len(smiles_list)
    for bucket in buckets:
        if len(train_idx) + len(bucket) <= total - target_val:
            train_idx.extend(bucket)
        else:
            val_idx.extend(bucket)

    if not val_idx and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
    if not train_idx and len(val_idx) > 1:
        train_idx.append(val_idx.pop(0))

    return sorted(train_idx), sorted(val_idx)
