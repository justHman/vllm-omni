"""Unit tests for the VieNeu-TTS-v2 pipeline YAML schema.

Loads ``vllm_omni/model_executor/models/vieneu/pipeline.yaml`` with
``yaml.safe_load`` and asserts the structural contract the integration
relies on: stage count, stage roles, final-output flags, connector chunk
config, and the talker stop-token id.

No production code is imported -- only ``pyyaml`` + stdlib -- so this file
runs without vLLM / aenum / neucodec / sea-g2p / torch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE_YAML = (
    Path(__file__).resolve().parent.parent
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "vieneu"
    / "pipeline.yaml"
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(PIPELINE_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_top_level_fields(cfg: dict):
    assert cfg["model_type"] == "vieneu"
    assert cfg["async_chunk"] is True


def test_exactly_two_stages(cfg: dict):
    stages = cfg["stages"]
    assert isinstance(stages, list)
    assert len(stages) == 2


def test_stage0_talker(cfg: dict):
    s0 = cfg["stages"][0]
    assert s0["model_stage"] == "vieneu_talker"
    assert s0["worker_type"] == "ar"
    engine_args = s0["engine_args"]
    assert engine_args["model_arch"] == "VieNeuTalkerForConditionalGeneration"


def test_stage0_stop_token_ids(cfg: dict):
    s0 = cfg["stages"][0]
    stop = s0["default_sampling_params"]["stop_token_ids"]
    assert stop == [381]


def test_stage1_codec(cfg: dict):
    s1 = cfg["stages"][1]
    assert s1["model_stage"] == "vieneu_codec"
    assert s1["worker_type"] == "generation"
    assert s1["final_output"] is True
    assert s1["final_output_type"] == "audio"


def test_connector_chunk_config(cfg: dict):
    connectors = cfg["connectors"]
    assert "connector_of_shared_memory" in connectors
    extra = connectors["connector_of_shared_memory"]["extra"]
    assert "codec_chunk_frames" in extra
    assert "codec_left_context_frames" in extra
