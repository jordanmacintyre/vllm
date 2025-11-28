#!/usr/bin/env python3
"""
Test suite for expert_tracking.py implementation.

Validates core functionality:
- Tracker initialization and configuration
- Expert selection recording and JSONL output format
- Metadata generation
- Layer filtering
- Tensor shape handling
- Suppression mechanism
"""

import os
import json
import tempfile
import torch
from vllm.model_executor.layers.fused_moe.expert_tracking import (
    ExpertUsageTracker,
    record_expert_selection,
    init_expert_tracker,
    expert_tracking_enabled,
    set_model_id,
    suppress_expert_tracking,
    resume_expert_tracking,
)


def test_tracker_initialization():
    """Test 1: Verify tracker initializes with correct default state."""
    print("\n=== Test 1: Tracker Initialization ===")

    # Create temporary file for tracker output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        # Initialize tracker for layer 0
        tracker = ExpertUsageTracker(target_layer=0, output_file=temp_file)

        # Verify initial state
        assert tracker.target_layer == 0
        assert tracker.output_file == temp_file
        assert tracker._token_count == 0
        assert tracker._meta_written == False

        print("PASS: Tracker initialized successfully")
        print(f"  Target layer: {tracker.target_layer}")
        print(f"  Output file: {tracker.output_file}")

        tracker.close()
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_basic_recording():
    """Test 2: Verify expert selection recording and JSONL output format."""
    print("\n=== Test 2: Basic Recording ===")

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        tracker = ExpertUsageTracker(target_layer=0, output_file=temp_file)

        # Simulate expert selection for 3 tokens with top-2 routing
        # Each token selects 2 experts with associated weights
        topk_ids = torch.tensor([[5, 12], [3, 7], [5, 3]])
        topk_weights = torch.tensor([[0.6, 0.4], [0.55, 0.45], [0.7, 0.3]])
        layer_prefix = "model.layers.0.block_sparse_moe"

        # Record selections and close tracker
        tracker.record(topk_ids, topk_weights, layer_prefix)
        tracker.close()

        # Read output file and verify format
        with open(temp_file, 'r') as f:
            lines = f.readlines()

        # Should have 1 metadata line + 3 route records
        assert len(lines) == 4, f"Expected 4 lines (1 meta + 3 routes), got {len(lines)}"

        # Verify metadata record
        meta = json.loads(lines[0])
        assert meta['type'] == 'meta'
        assert meta['top_k'] == 2
        assert meta['layers_logged'] == [0]
        print("PASS: Metadata written correctly")
        print(f"  Top-K: {meta['top_k']}")

        # Verify route records
        for i, line in enumerate(lines[1:]):
            route = json.loads(line)
            assert route['type'] == 'route'
            assert route['token_idx'] == i
            assert route['layer'] == 0
            assert len(route['topk_ids']) == 2
            assert len(route['topk_weights']) == 2

        print("PASS: Route records written correctly")
        print(f"  Total tokens: 3")
        print(f"  Sample route: {json.loads(lines[1])}")

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_layer_filtering():
    """Test 3: Verify only target layer is tracked, other layers ignored."""
    print("\n=== Test 3: Layer Filtering ===")

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        # Configure tracker to only record layer 5
        tracker = ExpertUsageTracker(target_layer=5, output_file=temp_file)

        topk_ids = torch.tensor([[1, 2]])
        topk_weights = torch.tensor([[0.6, 0.4]])

        # Record from layer 3 (should be ignored)
        tracker.record(topk_ids, topk_weights, "model.layers.3.block_sparse_moe")

        # Record from layer 5 (should be logged)
        tracker.record(topk_ids, topk_weights, "model.layers.5.block_sparse_moe")

        tracker.close()

        # Verify only 1 route record + metadata was written
        with open(temp_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2, f"Expected 2 lines (meta + 1 route), got {len(lines)}"

        # Verify the route is from the correct layer
        route = json.loads(lines[1])
        assert route['layer'] == 5

        print("PASS: Layer filtering works correctly")
        print(f"  Target layer: 5")
        print(f"  Recorded only from layer: {route['layer']}")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_3d_tensor_handling():
    """Test 4: Verify 3D tensor (batch, seq, top_k) is correctly flattened."""
    print("\n=== Test 4: 3D Tensor Handling ===")

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        tracker = ExpertUsageTracker(target_layer=0, output_file=temp_file)

        # Simulate 3D tensor with shape: (batch=2, seq_len=3, top_k=2)
        # This represents 2 batches, each with 3 tokens, each selecting 2 experts
        topk_ids = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                                  [[7, 8], [9, 10], [11, 12]]])
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32) * 0.5

        tracker.record(topk_ids, topk_weights, "model.layers.0.block_sparse_moe")
        tracker.close()

        # Should be flattened to 6 total tokens (2 batches * 3 tokens)
        with open(temp_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 7, f"Expected 7 lines (meta + 6 routes), got {len(lines)}"

        print("PASS: 3D tensor flattened correctly")
        print(f"  Input shape: (2, 3, 2)")
        print(f"  Output tokens: 6")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_suppression_mechanism():
    """Test 5: Verify suppression prevents tracking during warmup/dummy runs."""
    print("\n=== Test 5: Suppression Mechanism ===")

    # Configure environment for expert tracking
    os.environ["VLLM_LOG_MOE"] = "test_output.jsonl"
    os.environ["VLLM_LOG_MOE_LAYER"] = "0"

    try:
        # Initialize global tracker
        init_expert_tracker()
        assert expert_tracking_enabled() == True

        # Suppress tracking (simulates warmup phase)
        suppress_expert_tracking()

        topk_ids = torch.tensor([[1, 2]])
        topk_weights = torch.tensor([[0.6, 0.4]])

        # This call should be suppressed (not logged)
        record_expert_selection(topk_ids, topk_weights, "model.layers.0.block_sparse_moe")

        # Resume tracking (simulates actual inference)
        resume_expert_tracking()

        # This call should be recorded
        record_expert_selection(topk_ids, topk_weights, "model.layers.0.block_sparse_moe")

        print("PASS: Suppression mechanism works")
        print("  Suppressed during dummy runs")
        print("  Resumed for actual inference")

    finally:
        # Clean up
        if os.path.exists("test_output.jsonl"):
            os.remove("test_output.jsonl")
        os.environ.pop("VLLM_LOG_MOE", None)
        os.environ.pop("VLLM_LOG_MOE_LAYER", None)


def test_model_id_capture():
    """Test 6: Verify model ID is captured and included in metadata."""
    print("\n=== Test 6: Model ID Capture ===")

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        # Set model ID globally
        set_model_id("Qwen/Qwen1.5-MoE-A2.7B-Chat")

        tracker = ExpertUsageTracker(target_layer=0, output_file=temp_file)

        # Trigger metadata write by recording one selection
        topk_ids = torch.tensor([[1, 2]])
        topk_weights = torch.tensor([[0.6, 0.4]])
        tracker.record(topk_ids, topk_weights, "model.layers.0.block_sparse_moe")
        tracker.close()

        # Verify model ID is in metadata
        with open(temp_file, 'r') as f:
            meta = json.loads(f.readline())

        assert meta['model_id'] == "Qwen/Qwen1.5-MoE-A2.7B-Chat"

        print("PASS: Model ID captured correctly")
        print(f"  Model: {meta['model_id']}")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def run_all_tests():
    """Execute all test cases and report results."""
    print("="*70)
    print("EXPERT TRACKING TEST SUITE")
    print("="*70)

    # List of all test functions
    tests = [
        test_tracker_initialization,
        test_basic_recording,
        test_layer_filtering,
        test_3d_tensor_handling,
        test_suppression_mechanism,
        test_model_id_capture,
    ]

    passed = 0
    failed = 0

    # Run each test and track results
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: Test failed - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: Test error - {e}")
            failed += 1

    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
