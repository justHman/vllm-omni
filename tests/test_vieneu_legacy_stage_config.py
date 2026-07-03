"""Unit tests for the legacy VieNeu stage config used by ``vllm serve``.

The rc1 runtime path resolves stage configs via
``vllm_omni/model_executor/stage_configs/<model_type>.yaml`` and expects the
legacy ``stage_args:`` schema.  This test is deliberately pure-Python + PyYAML
(no vLLM import) so it catches the Colab failure mode where the forward-looking
``models/vieneu/pipeline.yaml`` existed but the serving path fell back to the
single-stage diffusion default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LEGACY_YAML = (
    Path(__file__).resolve().parent.parent
    / "vllm_omni"
    / "model_executor"
    / "stage_configs"
    / "vieneu.yaml"
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(LEGACY_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_legacy_schema_is_used_by_serve_path(cfg: dict):
    assert cfg["async_chunk"] is True
    assert "stage_args" in cfg
    assert "stages" not in cfg
    assert len(cfg["stage_args"]) == 2


def test_both_stages_are_llm_not_diffusion(cfg: dict):
    stage_args = cfg["stage_args"]
    assert [s["stage_type"] for s in stage_args] == ["llm", "llm"]


def test_stage0_talker_matches_vieneu_runtime(cfg: dict):
    s0 = cfg["stage_args"][0]
    ea = s0["engine_args"]
    assert s0["is_comprehension"] is True
    assert ea["model_stage"] == "vieneu_talker"
    assert ea["model_arch"] == "VieNeuTalkerForConditionalGeneration"
    assert ea["worker_type"] == "ar"
    assert ea["hf_overrides"]["architectures"] == ["VieNeuTalkerForConditionalGeneration"]
    assert s0["default_sampling_params"]["stop_token_ids"] == [381]


def test_stage1_codec_matches_vieneu_runtime(cfg: dict):
    s1 = cfg["stage_args"][1]
    ea = s1["engine_args"]
    assert ea["model_stage"] == "vieneu_codec"
    assert ea["model_arch"] == "VieNeuCodecDecoder"
    assert ea["worker_type"] == "generation"
    assert ea["hf_overrides"]["architectures"] == ["VieNeuCodecDecoder"]
    assert s1["engine_input_source"] == [0]
    assert s1["final_output"] is True
    assert s1["final_output_type"] == "audio"


def test_runtime_connector_config(cfg: dict):
    connector = cfg["runtime"]["connectors"]["connector_of_shared_memory"]
    assert connector["name"] == "SharedMemoryConnector"
    extra = connector["extra"]
    assert extra["codec_streaming"] is True
    assert extra["codec_chunk_frames"] == 25
    assert extra["codec_left_context_frames"] == 25
    assert cfg["runtime"]["edges"] == [{"from": 0, "to": 1, "window_size": -1}]
