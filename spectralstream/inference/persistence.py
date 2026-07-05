from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


class StateManager:
    def __init__(self, state_dir: str = "~/.spectralstream/state/"):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_interval: float = 300.0
        self._last_checkpoint: float = 0.0
        self._auto_save_enabled: bool = True
        self._subdirs = {
            "hdc": self.state_dir / "hdc",
            "gate": self.state_dir / "gate",
            "sessions": self.state_dir / "sessions",
            "checkpoints": self.state_dir / "checkpoints",
            "configs": self.state_dir / "configs",
        }
        for d in self._subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _dir_for(self, key: str) -> Path:
        return self._subdirs.get(key, self.state_dir)

    def _safe_path(self, subdir_key: str, name: str, ext: str) -> Path:
        safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._dir_for(subdir_key) / f"{safe_name}{ext}"

    def save_hdc_engine(self, engine, name: str = "default") -> bool:
        try:
            hd = engine.hd if hasattr(engine, "hd") else engine
            data = {
                "dim": hd.dim,
                "seed": hd.seed,
                "max_prototypes": hd.max_prototypes,
                "min_order": hd.min_order,
                "max_order": hd.max_order,
                "stopword_ids": list(hd.stopword_ids),
                "content_bias": hd.content_bias,
                "stopword_penalty": hd.stopword_penalty,
                "max_total_prototypes": hd.max_total_prototypes,
                "base_temperature": hd.base_temperature,
                "short_order_weight": hd.short_order_weight,
                "long_order_weight": hd.long_order_weight,
                "anomaly_threshold": hd.anomaly_threshold,
                "cache_max_size": hd.cache_max_size,
                "cache_hits": hd.cache_hits,
                "cache_misses": hd.cache_misses,
                "n_lsh_tables": len(hd.lsh_tables),
            }
            token_ids = sorted(hd.token_vectors.keys())
            tv_array = np.zeros((len(token_ids), hd.dim), dtype=np.int8)
            for i, tid in enumerate(token_ids):
                tv_array[i] = hd.token_vectors[tid]
            npz_path = self._safe_path("hdc", f"{name}_token_vectors", ".npz")
            np.savez_compressed(npz_path, vectors=tv_array, token_ids=token_ids)
            proto_data = []
            for ctx_str, plist in hd.prototypes.items():
                for hv, count, pid in plist:
                    proto_data.append(
                        {
                            "context": list(ctx_str),
                            "hv_bytes": hv.tobytes(),
                            "count": count,
                            "pid": pid,
                        }
                    )
            proto_path = self._safe_path("hdc", f"{name}_prototypes", ".json")
            with open(proto_path, "w") as f:
                json.dump(proto_data, f)
            lsh_data = []
            for table in hd.lsh_tables:
                entries = []
                for hval, keys in table.items():
                    entries.append({"hash": hval, "keys": [list(k) for k in keys]})
                lsh_data.append(entries)
            lsh_path = self._safe_path("hdc", f"{name}_lsh", ".json")
            with open(lsh_path, "w") as f:
                json.dump(lsh_data, f)
            proj_data = [p.tolist() for p in hd.lsh_projections]
            proj_path = self._safe_path("hdc", f"{name}_projections", ".json")
            with open(proj_path, "w") as f:
                json.dump(proj_data, f)
            meta = {
                "dim": hd.dim,
                "token_ids": token_ids,
                "n_prototypes": len(proto_data),
                "n_lsh_entries": sum(len(t) for t in lsh_data),
            }
            meta_path = self._safe_path("hdc", f"{name}_meta", ".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            if hasattr(engine, "ngram") and hasattr(engine.ngram, "counts"):
                ngram_data = {}
                for order, ct in enumerate(engine.ngram.counts):
                    order_str = str(order)
                    ngram_data[order_str] = {
                        " ".join(str(t) for t in ctx): dict(tok_counts)
                        for ctx, tok_counts in dict(ct).items()
                    }
                ngram_path = self._safe_path("hdc", f"{name}_ngram", ".json")
                with open(ngram_path, "w") as f:
                    json.dump(ngram_data, f)
            return True
        except Exception as exc:
            print(f"[StateManager] save_hdc_engine failed: {exc}")
            return False

    def load_hdc_engine(self, engine, name: str = "default") -> bool:
        try:
            hd = engine.hd if hasattr(engine, "hd") else engine
            meta_path = self._safe_path("hdc", f"{name}_meta", ".json")
            if not meta_path.exists():
                return False
            tv_path = self._safe_path("hdc", f"{name}_token_vectors", ".npz")
            if tv_path.exists():
                npz = np.load(tv_path)
                token_ids = npz["token_ids"].tolist()
                vectors = npz["vectors"]
                for i, tid in enumerate(token_ids):
                    hd.token_vectors[int(tid)] = vectors[i]
            proto_path = self._safe_path("hdc", f"{name}_prototypes", ".json")
            if proto_path.exists():
                with open(proto_path) as f:
                    proto_data = json.load(f)
                hd.prototypes.clear()
                hd.prototype_match_counts.clear()
                for entry in proto_data:
                    ctx = tuple(entry["context"])
                    hv = np.frombuffer(bytes(entry["hv_bytes"]), dtype=np.int8).reshape(
                        hd.dim
                    )
                    if ctx not in hd.prototypes:
                        hd.prototypes[ctx] = []
                    hd.prototypes[ctx].append((hv, entry["count"], entry["pid"]))
                    hd.prototype_match_counts[entry["pid"]] = 0
                hd.prototype_counter = (
                    max((e["pid"] for e in proto_data), default=0) + 1
                )
            lsh_path = self._safe_path("hdc", f"{name}_lsh", ".json")
            if lsh_path.exists():
                with open(lsh_path) as f:
                    lsh_data = json.load(f)
                hd.lsh_tables.clear()
                for table_entries in lsh_data:
                    table = {}
                    for entry in table_entries:
                        hval = entry["hash"]
                        keys = [tuple(k) for k in entry["keys"]]
                        table[hval] = keys
                    hd.lsh_tables.append(table)
            proj_path = self._safe_path("hdc", f"{name}_projections", ".json")
            if proj_path.exists():
                with open(proj_path) as f:
                    proj_data = json.load(f)
                hd.lsh_projections = [np.array(p, dtype=np.int8) for p in proj_data]
            if hasattr(engine, "ngram") and hasattr(engine.ngram, "counts"):
                ngram_path = self._safe_path("hdc", f"{name}_ngram", ".json")
                if ngram_path.exists():
                    with open(ngram_path) as f:
                        ngram_data = json.load(f)
                    for order_str, ctx_data in ngram_data.items():
                        order = int(order_str)
                        if order < len(engine.ngram.counts):
                            for ctx_str, tok_counts in ctx_data.items():
                                ctx = tuple(int(t) for t in ctx_str.split())
                                for tok_str, cnt in tok_counts.items():
                                    engine.ngram.counts[order][ctx][int(tok_str)] = cnt
            return True
        except Exception as exc:
            print(f"[StateManager] load_hdc_engine failed: {exc}")
            return False

    def save_confidence_gate(self, gate, name: str = "default") -> bool:
        try:
            data = {
                "weights": gate.weights.tolist(),
                "bias": gate.bias,
                "lr": gate.lr,
                "base_lr": gate.base_lr,
                "update_count": gate.update_count,
                "total_train": gate.total_train,
                "total_correct_pred": gate.total_correct_pred,
                "content_weights": getattr(gate, "content_weights", {}),
                "neg_preds": list(gate.neg_preds),
                "n_features": gate.n_features,
            }
            path = self._safe_path("gate", f"{name}", ".json")
            with open(path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as exc:
            print(f"[StateManager] save_confidence_gate failed: {exc}")
            return False

    def load_confidence_gate(self, gate, name: str = "default") -> bool:
        try:
            path = self._safe_path("gate", f"{name}", ".json")
            if not path.exists():
                return False
            with open(path) as f:
                data = json.load(f)
            gate.weights = np.array(data["weights"], dtype=np.float32)
            gate.bias = data["bias"]
            gate.lr = data["lr"]
            gate.base_lr = data["base_lr"]
            gate.update_count = data["update_count"]
            gate.total_train = data["total_train"]
            gate.total_correct_pred = data["total_correct_pred"]
            gate.content_weights = data.get("content_weights", {})
            gate.n_features = data.get("n_features", gate.n_features)
            gate.neg_preds = deque(data.get("neg_preds", []), maxlen=500)
            return True
        except Exception as exc:
            print(f"[StateManager] load_confidence_gate failed: {exc}")
            return False

    def save_online_learning(self, learner, name: str = "default") -> bool:
        try:
            data = {
                "total_corrections": learner.total_corrections,
                "total_acceptances": learner.total_acceptances,
                "recent_accuracy": list(learner.recent_accuracy),
                "max_buffer": learner.max_buffer,
            }
            path = self._safe_path("configs", f"{name}_online_learning", ".json")
            with open(path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as exc:
            print(f"[StateManager] save_online_learning failed: {exc}")
            return False

    def load_online_learning(self, learner, name: str = "default") -> bool:
        try:
            path = self._safe_path("configs", f"{name}_online_learning", ".json")
            if not path.exists():
                return False
            with open(path) as f:
                data = json.load(f)
            learner.total_corrections = data["total_corrections"]
            learner.total_acceptances = data["total_acceptances"]
            learner.recent_accuracy = deque(data.get("recent_accuracy", []), maxlen=100)
            learner.max_buffer = data.get("max_buffer", learner.max_buffer)
            return True
        except Exception as exc:
            print(f"[StateManager] load_online_learning failed: {exc}")
            return False

    def save_resonance_calibration(self, controller, name: str = "default") -> bool:
        try:
            data = {
                "block_size": controller.block_size,
                "temperature": controller.temperature,
                "coherence_threshold": controller.coherence_threshold,
                "n_candidates": controller.n_candidates,
                "error_integral": controller.error_integral,
                "prev_error": controller.prev_error,
                "target": controller.target,
                "adaptation_count": controller.adaptation_count,
            }
            path = self._safe_path("configs", f"{name}_resonance", ".json")
            with open(path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as exc:
            print(f"[StateManager] save_resonance_calibration failed: {exc}")
            return False

    def load_resonance_calibration(self, controller, name: str = "default") -> bool:
        try:
            path = self._safe_path("configs", f"{name}_resonance", ".json")
            if not path.exists():
                return False
            with open(path) as f:
                data = json.load(f)
            controller.block_size = data["block_size"]
            controller.temperature = data["temperature"]
            controller.coherence_threshold = data["coherence_threshold"]
            controller.n_candidates = data["n_candidates"]
            controller.error_integral = data["error_integral"]
            controller.prev_error = data["prev_error"]
            controller.target = data["target"]
            controller.adaptation_count = data["adaptation_count"]
            return True
        except Exception as exc:
            print(f"[StateManager] load_resonance_calibration failed: {exc}")
            return False

    def save_session(self, session_name: str, kv_cache, stats) -> bool:
        try:
            session_dir = self._subdirs["sessions"] / session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            if kv_cache is not None:
                if hasattr(kv_cache, "keys") and hasattr(kv_cache, "values"):
                    k_arr = np.array(
                        kv_cache.keys()
                        if callable(kv_cache.keys)
                        else list(kv_cache.keys())
                    )
                    v_arr = np.array(
                        kv_cache.values()
                        if callable(kv_cache.values)
                        else list(kv_cache.values())
                    )
                elif hasattr(kv_cache, "cache_k") and hasattr(kv_cache, "cache_v"):
                    k_arr = kv_cache.cache_k
                    v_arr = kv_cache.cache_v
                else:
                    k_arr = np.array([])
                    v_arr = np.array([])
                np.savez_compressed(
                    session_dir / "kv_cache.npz",
                    keys=k_arr,
                    values=v_arr,
                )
            meta = {
                "timestamp": time.time(),
                "stats": stats if stats else {},
                "session_name": session_name,
            }
            with open(session_dir / "meta.json", "w") as f:
                json.dump(meta, f)
            return True
        except Exception as exc:
            print(f"[StateManager] save_session failed: {exc}")
            return False

    def load_session(self, session_name: str) -> bool:
        try:
            session_dir = self._subdirs["sessions"] / session_name
            if not session_dir.exists():
                return False
            meta_file = session_dir / "meta.json"
            if not meta_file.exists():
                return False
            return True
        except Exception as exc:
            print(f"[StateManager] load_session check failed: {exc}")
            return False

    def restore_session_kv(self, session_name: str):
        try:
            session_dir = self._subdirs["sessions"] / session_name
            kv_file = session_dir / "kv_cache.npz"
            if not kv_file.exists():
                return None
            npz = np.load(kv_file)
            return {
                "keys": npz.get("keys"),
                "values": npz.get("values"),
            }
        except Exception as exc:
            print(f"[StateManager] restore_session_kv failed: {exc}")
            return None

    def restore_session_meta(self, session_name: str) -> Optional[dict]:
        try:
            session_dir = self._subdirs["sessions"] / session_name
            meta_file = session_dir / "meta.json"
            if not meta_file.exists():
                return None
            with open(meta_file) as f:
                return json.load(f)
        except Exception as exc:
            print(f"[StateManager] restore_session_meta failed: {exc}")
            return None

    def list_sessions(self) -> list[str]:
        try:
            session_dir = self._subdirs["sessions"]
            if not session_dir.exists():
                return []
            return sorted(
                d.name
                for d in session_dir.iterdir()
                if d.is_dir() and (d / "meta.json").exists()
            )
        except Exception as exc:
            print(f"[StateManager] list_sessions failed: {exc}")
            return []

    def delete_session(self, session_name: str) -> bool:
        try:
            session_dir = self._subdirs["sessions"] / session_name
            if session_dir.exists():
                shutil.rmtree(session_dir)
                return True
            return False
        except Exception as exc:
            print(f"[StateManager] delete_session failed: {exc}")
            return False

    def save_checkpoint(self) -> bool:
        try:
            ckpt_dir = self._subdirs["checkpoints"]
            timestamp = int(time.time())
            ckpt_name = f"checkpoint_{timestamp}"
            ckpt_path = ckpt_dir / ckpt_name
            ckpt_path.mkdir(exist_ok=True)
            meta = {
                "timestamp": timestamp,
                "type": "full_checkpoint",
                "components_saved": [],
            }
            for src_dir_key in ["hdc", "gate", "configs"]:
                src = self._dir_for(src_dir_key)
                dst = ckpt_path / src_dir_key
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    meta["components_saved"].append(src_dir_key)
            with open(ckpt_path / "checkpoint_meta.json", "w") as f:
                json.dump(meta, f)
            self._last_checkpoint = time.time()
            self._prune_old_checkpoints()
            return True
        except Exception as exc:
            print(f"[StateManager] save_checkpoint failed: {exc}")
            return False

    def _prune_old_checkpoints(self, max_keep: int = 10):
        try:
            ckpt_dir = self._subdirs["checkpoints"]
            checkpoints = sorted(d for d in ckpt_dir.iterdir() if d.is_dir())
            while len(checkpoints) > max_keep:
                oldest = checkpoints.pop(0)
                shutil.rmtree(oldest)
        except Exception:
            pass

    def list_checkpoints(self) -> list[str]:
        try:
            ckpt_dir = self._subdirs["checkpoints"]
            if not ckpt_dir.exists():
                return []
            return sorted(
                d.name
                for d in ckpt_dir.iterdir()
                if d.is_dir() and (d / "checkpoint_meta.json").exists()
            )
        except Exception:
            return []

    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        try:
            ckpt_path = self._subdirs["checkpoints"] / checkpoint_name
            if not ckpt_path.exists():
                return False
            for src_dir_key in ["hdc", "gate", "configs"]:
                src = ckpt_path / src_dir_key
                if src.exists():
                    dst = self._dir_for(src_dir_key)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            return True
        except Exception as exc:
            print(f"[StateManager] restore_checkpoint failed: {exc}")
            return False

    def maybe_checkpoint(self, force: bool = False) -> bool:
        if force:
            return self.save_checkpoint()
        elapsed = time.time() - self._last_checkpoint
        if self._auto_save_enabled and elapsed >= self._checkpoint_interval:
            return self.save_checkpoint()
        return False

    def auto_save(
        self,
        engine=None,
        gate=None,
        learner=None,
        controller=None,
        name: str = "default",
    ) -> bool:
        success = True
        if engine is not None:
            success &= self.save_hdc_engine(engine, name=name)
        if gate is not None:
            success &= self.save_confidence_gate(gate, name=name)
        if learner is not None:
            success &= self.save_online_learning(learner, name=name)
        if controller is not None:
            success &= self.save_resonance_calibration(controller, name=name)
        return success

    def auto_load(
        self,
        engine=None,
        gate=None,
        learner=None,
        controller=None,
        name: str = "default",
    ) -> bool:
        success = True
        if engine is not None:
            if not self.load_hdc_engine(engine, name=name):
                success = False
        if gate is not None:
            if not self.load_confidence_gate(gate, name=name):
                success = False
        if learner is not None:
            if not self.load_online_learning(learner, name=name):
                success = False
        if controller is not None:
            if not self.load_resonance_calibration(controller, name=name):
                success = False
        return success

    def save_config(self, config: dict, name: str = "default") -> bool:
        try:
            path = self._safe_path("configs", f"{name}_config", ".json")
            with open(path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except Exception as exc:
            print(f"[StateManager] save_config failed: {exc}")
            return False

    def load_config(self, name: str = "default") -> Optional[dict]:
        try:
            path = self._safe_path("configs", f"{name}_config", ".json")
            if not path.exists():
                return None
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            print(f"[StateManager] load_config failed: {exc}")
            return None

    def clear(self):
        for d in self._subdirs.values():
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
