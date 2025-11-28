"""
Expert Usage Tracking for vLLM MoE Models

Tracks which experts are selected for each token during inference.
Outputs per-token expert selections to a JSONL file.

Environment Variables:
    VLLM_LOG_MOE: Set to an output file path to enable tracking 
    (e.g., "expert_log.jsonl") If not set, tracking is disabled.
    VLLM_EXPERT_TRACKING_LAYER: Layer index to track (0-indexed, default: 0)

Integration Point:
    Call record_expert_selection() in FusedMoEMethodBase (or subclass)
    after select_experts() and before fused_experts().
"""

import os
import re
import json
import atexit
from typing import Optional, List

import torch

# ============================================================================
# Configuration
# ============================================================================

# Check if user wants to log expert usage
_OUTPUT_FILE = os.environ.get("VLLM_LOG_MOE")
EXPERT_TRACKING_ENABLED = _OUTPUT_FILE is not None and _OUTPUT_FILE != ""

# Which layer to track (default: layer 0)
_TARGET_LAYER_STR = os.environ.get("VLLM_LOG_MOE_LAYER", "0")

# Global tracker instance + suppression flag
_tracker: Optional["ExpertUsageTracker"] = None

# Flag to suppress tracking during warmup/profiling
_SUPPRESS_TRACKING: bool = False

print(
    f"[ExpertTracker] module import: enabled={EXPERT_TRACKING_ENABLED} "
    f"layer={_TARGET_LAYER_STR} file={_OUTPUT_FILE}"
)

# ==============================================================================
# Control Functions
# ==============================================================================


def suppress_tracking() -> None:
    """Disable logging during dummy/profile/CUDA-graph runs."""
    global _SUPPRESS_TRACKING
    _SUPPRESS_TRACKING = True


def resume_tracking() -> None:
    """Re-enable logging after dummy/profile/CUDA-graph runs."""
    global _SUPPRESS_TRACKING
    _SUPPRESS_TRACKING = False


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_layer_from_prefix(prefix: str) -> Optional[int]:
    """
    Extract layer number from layer name string.

    Examples:
        "model.layers.5.block_sparse_moe" -> 5
        "model.layers.12.mlp.experts" -> 12
        "transformer.h.7.moe" -> 7
    """
    # Common patterns: "layers.N." or ".h.N."
    match = re.search(r"(?:layers|\.h)\.(\d+)\.", prefix)
    if match:
        return int(match.group(1))
    return None


def _get_vllm_version():
    """Get vLLM version string."""
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _get_device_info():
    """Get current CUDA device info."""
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
        except Exception:
            return "cuda"
    return "cpu"


# ============================================================================
# Tracker Class
# ============================================================================


class ExpertUsageTracker:
    """
    Logs expert selections to a JSONL file.

    Output format (one JSON per line):
        Line 1: {"type": "meta", "model_id": "...", "top_k": 2, ...}
        Line 2+: {"type": "route", "token_idx": 0, "topk_ids": [3, 7], ...}
    """

    def __init__(self, target_layer: int, output_file: str):
        # Configuration
        self.target_layer = target_layer
        self.output_file = output_file

        # File handle
        self._file = None

        # Counters
        self._token_count = 0  # Total tokens logged
        self._write_count = 0  # Number of writes since last flush
        self._call_count = 0  # Number of record() calls

        # Metadata tracking
        self._meta_written = False
        self._top_k: Optional[int] = None  # Inferred from first batch

        # System info for metadata
        self.model_id: str = "unknown"
        self.vllm_version: str = _get_vllm_version()
        self.torch_version: str = torch.__version__
        self.device: str = _get_device_info()
        self.layers_logged: List[int] = [self.target_layer]

    def _open_log_file(self) -> None:
        """Open the output file for writing."""
        if self._file is None:
            self._file = open(self.output_file, "w")
            print(
                f"[ExpertTracker] Tracking layer {self.target_layer} "
                f"-> {self.output_file}"
            )

    def _write_meta_if_needed(self) -> None:
        """Write the metadata record (first line of file)."""
        if self._meta_written:
            return

        top_k = int(self._top_k) if self._top_k is not None else 0

        # Create metadata record
        meta = {
            "type": "meta",
            "model_id": self.model_id,
            "vllm_version": self.vllm_version,
            "torch_version": self.torch_version,
            "device": self.device,
            "layers_logged": self.layers_logged,
            "top_k": top_k,
        }

        # Write to file
        self._open_log_file()
        self._file.write(json.dumps(meta) + "\n")
        self._meta_written = True

    def record(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor, layer_prefix: str
    ) -> None:
        """
        Record expert selections for a batch of tokens.

        Args:
            topk_ids: Tensor of shape [num_tokens, top_k] with expert IDs
            topk_weights: Tensor of shape [num_tokens, top_k] with routing weights
            layer_prefix: String like "model.layers.0.block_sparse_moe"
        """
        # Check if this is the layer we're tracking
        layer_idx = _extract_layer_from_prefix(layer_prefix)
        if layer_idx is None or layer_idx != self.target_layer:
            return

        self._call_count += 1

        # Handle 3D tensors (batch, seq, top_k) -> flatten to (tokens, top_k)
        if topk_ids.dim() == 3:
            topk_ids = topk_ids.reshape(-1, topk_ids.size(-1))
            topk_weights = topk_weights.reshape(-1, topk_weights.size(-1))

        num_tokens, k = topk_ids.shape

        # Infer top_k once for meta
        if self._top_k is None:
            self._top_k = int(k)

        # Write metadata before first route record
        self._write_meta_if_needed()

        # Move tensors to CPU
        ids_cpu = topk_ids.detach().cpu()
        weights_cpu = topk_weights.detach().cpu()

        # Request ID for grouping tokens from the same forward pass
        req_id = f"call_{self._call_count}"

        # Write one record per token
        for i in range(num_tokens):
            rec = {
                "type": "route",
                "req_id": req_id,
                "token_idx": int(self._token_count),
                "layer": int(layer_idx),
                "topk_ids": ids_cpu[i].tolist(),
                "topk_weights": [round(float(w), 6) for w in weights_cpu[i].tolist()],
            }
            self._file.write(json.dumps(rec) + "\n")

            self._token_count += 1
            self._write_count += 1

            # Flush periodically to avoid losing data on crashes
            if self._write_count >= 1000:
                self._file.flush()
                self._write_count = 0

    def close(self) -> None:
        """Close the output file and print summary."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[ExpertTracker] Wrote {self._token_count} token records")


# ============================================================================
# Public API
# ============================================================================


def init_tracker() -> None:
    """Initialize the global tracker instance."""
    global _tracker

    # Already initialized
    if _tracker is not None:
        return

    # Tracking disabled
    if not EXPERT_TRACKING_ENABLED:
        return

    # Parse target layer
    try:
        target_layer = int(_TARGET_LAYER_STR)
    except ValueError:
        print(
            f"[ExpertTracker] Invalid VLLM_EXPERT_TRACKING_LAYER: {_TARGET_LAYER_STR}"
        )
        return

    # Create tracker and register cleanup
    _tracker = ExpertUsageTracker(target_layer, _OUTPUT_FILE)
    atexit.register(_tracker.close)
    print(f"[ExpertTracker] Initialized for layer {_tracker.target_layer}")


def get_tracker() -> Optional[ExpertUsageTracker]:
    """Get the global tracker instance (may be None)."""
    return _tracker


def record_expert_selection(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_prefix: str,
) -> None:
    """
    Main entry point: record expert selections for a batch of tokens.

    Call this from your MoE layer after select_experts() returns topk_ids
    and topk_weights, before calling fused_experts().

    Args:
        topk_ids: Tensor [num_tokens, top_k] with selected expert IDs
        topk_weights: Tensor [num_tokens, top_k] with routing weights
        layer_prefix: String like "model.layers.0.block_sparse_moe"
    """
    global _tracker

    # Tracking disabled
    if not EXPERT_TRACKING_ENABLED:
        return

    # Suppressed (warmup/profiling)
    if _SUPPRESS_TRACKING:
        return

    # Lazy initialization
    if _tracker is None:
        init_tracker()
        if _tracker is None:
            return
        
    # Record data
    _tracker.record(topk_ids, topk_weights, layer_prefix)
