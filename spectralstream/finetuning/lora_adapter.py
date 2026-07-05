"""
LoRA Adapter for Compressed Model Fine-Tuning
==============================================
Low-Rank Adaptation: y = Wx + BAx
  B in R^{d_out x r}, A in R^{r x d_in}
Adapter merging: W_new = W + BA
"""

import json
import math
import os
from typing import Optional

import numpy as np


class LoRAAdapterV2:
    """Low-Rank Adaptation adapter for a single linear layer."""

    def __init__(self, d_in, d_out, r=8, alpha=16.0, dropout=0.05):
        self.r = r
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = dropout

        scale = 1.0 / math.sqrt(r)
        self.A = np.random.randn(r, d_in).astype(np.float32) * scale
        self.B = np.zeros((d_out, r), dtype=np.float32)

        self.A_grad = np.zeros_like(self.A)
        self.B_grad = np.zeros_like(self.B)
        self._step_count = 0
        self._training = False

    def forward(self, x):
        """y = Wx + BAx. Returns the LoRA delta only: scaling * B @ A @ x"""
        h = self.A @ x
        if self.dropout > 0 and self._training:
            mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float32)
            h = h * mask / (1 - self.dropout)
        return self.scaling * (self.B @ h)

    def merge_into(self, W):
        """W_new = W + scaling * B @ A"""
        return W + self.scaling * (self.B @ self.A)

    def backward(self, x, grad_output):
        """Compute gradients for A and B given input x and grad w.r.t. output."""
        h = self.A @ x
        grad_B = self.scaling * np.outer(grad_output, h)
        grad_A = self.scaling * np.outer(self.B.T @ grad_output, x)
        return grad_A, grad_B

    def step(self, lr=1e-4, weight_decay=0.01, grad_norm=1.0):
        """Apply gradient update with clipping and weight decay."""
        if grad_norm > 0:
            total_norm = math.sqrt(
                np.sum(self.A_grad**2) + np.sum(self.B_grad**2) + 1e-8
            )
            scale = min(grad_norm / total_norm, 1.0)
            self.A_grad *= scale
            self.B_grad *= scale
        self.A -= lr * (self.A_grad + weight_decay * self.A)
        self.B -= lr * (self.B_grad + weight_decay * self.B)
        self.A_grad = np.zeros_like(self.A)
        self.B_grad = np.zeros_like(self.B)

    def accumulate(self, grad_A, grad_B):
        self.A_grad += grad_A
        self.B_grad += grad_B
        self._step_count += 1

    @property
    def num_params(self):
        return self.A.size + self.B.size

    def save(self, path):
        np.savez(
            path,
            A=self.A,
            B=self.B,
            metadata=np.array([self.r, self.d_in, self.d_out, self.alpha]),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        meta = data["metadata"]
        r, d_in, d_out, alpha = int(meta[0]), int(meta[1]), int(meta[2]), float(meta[3])
        adapter = cls(d_in, d_out, r, alpha)
        adapter.A = data["A"]
        adapter.B = data["B"]
        return adapter


class CompressedLoRA:
    """Manages LoRA adapters for all target modules of a compressed model."""

    def __init__(self, rank=16, alpha=32, dropout=0.05, target_modules=None):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or []
        self.adapters = {}
        self._training = True

    def add_adapter(self, name, d_in, d_out, rank=None):
        r = rank or self.rank
        adapter = LoRAAdapterV2(d_in, d_out, r, self.alpha, self.dropout)
        adapter._training = self._training
        self.adapters[name] = adapter
        return adapter

    def forward(self, name, x):
        if name not in self.adapters:
            return np.zeros(
                (self.adapters[list(self.adapters.keys())[0]].d_out, x.shape[1])
                if self.adapters
                else (0,)
            )
        return self.adapters[name].forward(x)

    def merge_all(self, weight_dict):
        """Merge LoRA adapters into weight dict. Returns new weight dict."""
        merged = dict(weight_dict)
        for name, adapter in self.adapters.items():
            if name in merged:
                merged[name] = adapter.merge_into(merged[name])
        return merged

    def train_mode(self):
        self._training = True
        for a in self.adapters.values():
            a._training = True

    def eval_mode(self):
        self._training = False
        for a in self.adapters.values():
            a._training = False

    def save_all(self, directory):
        os.makedirs(directory, exist_ok=True)
        manifest = {"rank": self.rank, "alpha": self.alpha, "adapters": {}}
        for name, adapter in self.adapters.items():
            fname = name.replace(".", "_") + ".npz"
            adapter.save(os.path.join(directory, fname))
            manifest["adapters"][name] = {
                "file": fname,
                "d_in": adapter.d_in,
                "d_out": adapter.d_out,
                "r": adapter.r,
            }
        with open(os.path.join(directory, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load_all(cls, directory):
        with open(os.path.join(directory, "manifest.json")) as f:
            manifest = json.load(f)
        lora = cls(rank=manifest["rank"], alpha=manifest["alpha"])
        for name, info in manifest["adapters"].items():
            adapter = LoRAAdapterV2.load(os.path.join(directory, info["file"]))
            lora.adapters[name] = adapter
        return lora

    @property
    def total_params(self):
        return sum(a.num_params for a in self.adapters.values())

    @property
    def summary(self):
        lines = [
            f"CompressedLoRA: {len(self.adapters)} adapters, {self.total_params} params"
        ]
        for name, a in self.adapters.items():
            lines.append(
                f"  {name}: r={a.r} ({a.d_in}x{a.d_out}) = {a.num_params} params"
            )
        return chr(10).join(lines)


# Backward-compat alias — tests import LoRAAdapter from lora_adapter
LoRAAdapter = LoRAAdapterV2
