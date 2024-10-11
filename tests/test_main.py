import pytest
from pathlib import Path
from coremlprofiler import CoreMLProfiler
from huggingface_hub import snapshot_download


def test_integration_with_real_model():
    repo_path = snapshot_download(repo_id="FL33TW00D-HF/test-st", local_dir="./")
    model_path = repo_path + "/sentence_transformer_all-MiniLM-L6-v2.mlpackage"

    profiler = CoreMLProfiler(model_path)
    print(profiler.device_usage_summary())
