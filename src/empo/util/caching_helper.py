"""Caching helper for EMPO backward induction artifacts.

Caches DAGs, Phase 1 (human policy prior), and Phase 2 (robot policy) results
locally on disk and on Hugging Face Hub, keyed by world name, max_steps, and
hyperparameters.

Usage:
    from empo.util.caching_helper import CacheHelper

    cache = CacheHelper("trivial", max_steps=10)
    dag = cache.load_or_compute("dag", lambda: world_model.get_dag(return_probabilities=True))
    policy = cache.load_or_compute("phase2", lambda: compute_robot_policy(...),
                                    params=dict(beta_r=10.0, gamma_r=1.0))
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DEFAULT_LOCAL_DIR = os.path.expanduser("~/.empo_cache")
DEFAULT_HF_REPO = "EMPO-RL/empo-cache"
def _param_hash(params: Dict[str, Any]) -> str:
    """Generate a short deterministic hash from a parameter dict.

    Sorts keys and serializes to JSON so that identical parameters always
    produce the same hash, regardless of dict insertion order.

    Returns the first 10 hex chars of the SHA-256 digest.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def _cache_path(world_name: str, max_steps: int, artifact_type: str,
                params: Optional[Dict[str, Any]] = None) -> str:
    """Build a relative cache path for an artifact.

    Returns:
        A relative path string like "trivial/10/dag.pkl"
        or "trivial/10/phase2_a3b2c1d4e5.pkl"
    """
    base = f"{world_name}/{max_steps}"
    if params:
        h = _param_hash(params)
        return f"{base}/{artifact_type}_{h}.pkl"
    return f"{base}/{artifact_type}.pkl"
class CacheHelper:
    """Cache manager for EMPO backward induction artifacts.

    Looks up artifacts in this order:
    1. Local cache directory (~/.empo_cache by default)
    2. Hugging Face Hub (EMPO-RL/empo-cache by default)
    3. Compute from scratch via user-provided callable, then save

    Args:
        world_name: Name of the world/environment (e.g. "trivial", "basic/two_agents").
                    Should match the yaml filename without extension.
        max_steps:  Maximum steps for the environment. Part of the cache key
                    since different horizons produce different DAGs.
        local_dir:  Local cache directory. Default: ~/.empo_cache
        hf_repo:    Hugging Face Hub repo ID. Default: EMPO-RL/empo-cache
        auto_upload: Automatically upload newly computed artifacts to HF Hub.
    """

    def __init__(
        self,
        world_name: str,
        max_steps: int,
        local_dir: str = DEFAULT_LOCAL_DIR,
        hf_repo: str = DEFAULT_HF_REPO,
        auto_upload: bool = False,
    ):
        self.world_name = world_name
        self.max_steps = max_steps
        self.local_dir = Path(local_dir)
        self.hf_repo = hf_repo
        self.auto_upload = auto_upload

    def cache_path(self, artifact_type: str,
                   params: Optional[Dict[str, Any]] = None) -> str:
        """Get the relative cache path for an artifact (used as HF Hub filename too)."""
        return _cache_path(self.world_name, self.max_steps, artifact_type, params)

    def local_path(self, artifact_type: str,
                   params: Optional[Dict[str, Any]] = None) -> Path:
        """Get the full local filesystem path for an artifact."""
        return self.local_dir / self.cache_path(artifact_type, params)

    # ── Local cache operations ──────────────────────────────────────────

    def load_local(self, artifact_type: str,
                   params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Try to load an artifact from local cache.

        Returns the unpickled object, or None if not found.
        """
        path = self.local_path(artifact_type, params)
        if path.exists():
            print(f"✓ Loading cached {artifact_type} from {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save_local(self, obj: Any, artifact_type: str,
                   params: Optional[Dict[str, Any]] = None) -> Path:
        """Save an artifact to local cache.

        Also writes a .json sidecar with the params dict for human readability.
        Returns the path where the .pkl was saved.
        """
        path = self.local_path(artifact_type, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving {artifact_type} to {path}")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Write sidecar JSON with params (so humans can inspect cache contents)
        if params:
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump({"artifact_type": artifact_type,
                           "world_name": self.world_name,
                           "max_steps": self.max_steps,
                           "params": params}, f, indent=2, default=str)
        return path

    # ── Hugging Face Hub operations ─────────────────────────────────────

    def load_from_hf(self, artifact_type: str,
                     params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Try to download an artifact from Hugging Face Hub.

        If found, also saves to local cache for future use.
        Returns the unpickled object, or None if not found / on error.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("⚠ huggingface_hub not installed — skipping HF Hub lookup")
            return None

        rel_path = self.cache_path(artifact_type, params)
        try:
            print(f"🔍 Looking for {rel_path} on HF Hub ({self.hf_repo})...")
            local_file = hf_hub_download(
                repo_id=self.hf_repo,
                filename=rel_path,
                repo_type="dataset",
            )
            print(f"✓ Downloaded from HF Hub")
            with open(local_file, "rb") as f:
                obj = pickle.load(f)
            # Cache locally so we don't download again
            self.save_local(obj, artifact_type, params)
            return obj
        except Exception as e:
            print(f"✗ Not found on HF Hub: {e}")
            return None

    def upload_to_hf(self, artifact_type: str,
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """Upload a locally cached artifact to Hugging Face Hub.

        The artifact must already exist in the local cache.
        Returns True on success, False on failure.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("⚠ huggingface_hub not installed — cannot upload")
            return False

        local = self.local_path(artifact_type, params)
        if not local.exists():
            print(f"✗ No local file at {local} — nothing to upload")
            return False

        rel_path = self.cache_path(artifact_type, params)
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=rel_path,
                repo_id=self.hf_repo,
                repo_type="dataset",
            )
            # Also upload the sidecar JSON if it exists
            json_local = local.with_suffix(".json")
            if json_local.exists():
                api.upload_file(
                    path_or_fileobj=str(json_local),
                    path_in_repo=rel_path.replace(".pkl", ".json"),
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                )
            print(f"✓ Uploaded {rel_path} to {self.hf_repo}")
            return True
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False

    # ── Main entry point ────────────────────────────────────────────────

    def load_or_compute(
        self,
        artifact_type: str,
        compute_fn: Callable[[], Any],
        params: Optional[Dict[str, Any]] = None,
        skip_hf: bool = False,
    ) -> Any:
        """Load a cached artifact, or compute and cache it.

        Lookup order:
            1. Local cache
            2. HF Hub (unless skip_hf=True)
            3. Compute via compute_fn(), then save locally (and upload if auto_upload)

        Args:
            artifact_type: Label for this artifact, e.g. "dag", "phase1", "phase2".
            compute_fn:    Zero-argument callable that produces the artifact.
            params:        Dict of hyperparameters that affect this artifact.
                           Used to generate a unique cache key. Omit for DAG
                           (which only depends on world_name + max_steps).
            skip_hf:       If True, don't try HF Hub (useful offline).

        Returns:
            The cached or freshly computed artifact.
        """
        # 1. Try local
        result = self.load_local(artifact_type, params)
        if result is not None:
            return result

        # 2. Try HF Hub
        if not skip_hf:
            result = self.load_from_hf(artifact_type, params)
            if result is not None:
                return result

        # 3. Compute from scratch
        print(f"⏳ Computing {artifact_type} from scratch...")
        result = compute_fn()

        # Save locally
        self.save_local(result, artifact_type, params)

        # Optionally upload to HF Hub
        if self.auto_upload:
            self.upload_to_hf(artifact_type, params)

        return result

    # ── Utilities ───────────────────────────────────────────────────────

    def list_local(self) -> list:
        """List all locally cached .pkl files for this world/steps combination."""
        cache_dir = self.local_dir / self.world_name / str(self.max_steps)
        if not cache_dir.exists():
            return []
        return sorted(str(p.relative_to(self.local_dir))
                       for p in cache_dir.glob("*.pkl"))

    def clear_local(self, artifact_type: Optional[str] = None,
                    params: Optional[Dict[str, Any]] = None) -> int:
        """Clear local cache files.

        If artifact_type is given, clear just that one artifact.
        Otherwise, clear ALL cached artifacts for this world/steps.
        Returns the number of files deleted.
        """
        if artifact_type:
            path = self.local_path(artifact_type, params)
            count = 0
            for p in [path, path.with_suffix(".json")]:
                if p.exists():
                    p.unlink()
                    count += 1
            if count:
                print(f"🗑 Deleted {path}")
            return min(count, 1)

        cache_dir = self.local_dir / self.world_name / str(self.max_steps)
        if not cache_dir.exists():
            return 0
        count = 0
        for p in cache_dir.glob("*"):
            p.unlink()
            count += 1
        print(f"🗑 Deleted {count} files from {cache_dir}")
        return count

    def __repr__(self) -> str:
        return (f"CacheHelper(world_name={self.world_name!r}, "
                f"max_steps={self.max_steps}, "
                f"local_dir={str(self.local_dir)!r}, "
                f"hf_repo={self.hf_repo!r})")
