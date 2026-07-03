"""Configuration classes for VieNeu-TTS-v2.

The upstream checkpoint (``pnnbao-ump/VieNeu-TTS-v2``) ships a plain Qwen3
``config.json`` (``model_type="qwen3"``, ``architectures=["Qwen3ForCausalLM"]``)
-- see docs/Architecture.md Part B.2. That means the checkpoint's own config
is not distinguishable from a generic Qwen3 causal LM, and pipeline
resolution for it relies on a tokenizer-vocabulary heuristic
(``StageConfigFactory._looks_like_vieneu_tts``) rather than ``model_type``
alone.

``VieNeuTalkerConfig`` below is what the ``vieneu`` pipeline stage actually
loads (via ``hf_overrides={"model_type": "vieneu_talker", ...}`` in
``pipeline.yaml``) -- it is a thin wrapper around the checkpoint's own Qwen3
fields plus the handful of VieNeu-specific token ids needed at inference
time (stop token, speech-token vocabulary range). No field renaming is
needed here, unlike ``FishSpeechSlowARConfig``, because VieNeu's
``config.json`` already uses standard Transformers/Qwen3 attribute names.
"""

from __future__ import annotations

from transformers import PretrainedConfig

# From the checkpoint's tokenizer (docs/Architecture.md Part B.3):
#   375                 <|endoftext|>                (tokenizer default EOS/PAD/UNK)
#   380                 <|SPEECH_GENERATION_START|>
#   381                 <|SPEECH_GENERATION_END|>     (actual inference-time stop token)
#   382 .. 65917        <|speech_0|> .. <|speech_65535|>   (NeuCodec FSQ codes)
#   65918 .. 66917      <|speaker_0|> .. <|speaker_999|>   (unused by the v2 SDK)
#   66918 .. 66937      <|emotion_0|> .. <|emotion_19|>    (only 0-3 confirmed used)
_DEFAULT_SPEECH_GENERATION_START_ID = 380
_DEFAULT_SPEECH_GENERATION_END_ID = 381
_DEFAULT_SPEECH_TOKEN_ID_START = 382
_DEFAULT_SPEECH_TOKEN_ID_END = 65918  # exclusive
_DEFAULT_TEXT_PROMPT_START_ID = 377
_DEFAULT_TEXT_PROMPT_END_ID = 378


class VieNeuTalkerConfig(PretrainedConfig):
    """Talker-stage config -- a standard Qwen3-shaped transformer.

    Field names already match ``vllm.model_executor.models.qwen3.Qwen3Model``
    (``hidden_size``, ``num_attention_heads``, ``num_hidden_layers``, ...)
    because the upstream checkpoint's ``config.json`` uses vanilla Qwen3
    naming -- unlike Fish Speech, no remapping layer is required. This class
    exists to (a) give the stage a distinct ``model_type`` for the
    ``hf_overrides`` mechanism (see ``pipeline.yaml``), and (b) carry the
    VieNeu-specific token ids that ``VieNeuTalkerForConditionalGeneration``
    needs for stop-token override and codec-logit masking.
    """

    model_type = "vieneu_talker"

    def __init__(
        self,
        vocab_size: int = 66938,
        hidden_size: int = 768,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 22,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        # VieNeu-specific token ids (defaults from the v2 checkpoint's
        # tokenizer; overridable for other checkpoints/vocab layouts).
        speech_generation_start_id: int = _DEFAULT_SPEECH_GENERATION_START_ID,
        speech_generation_end_id: int = _DEFAULT_SPEECH_GENERATION_END_ID,
        speech_token_id_start: int = _DEFAULT_SPEECH_TOKEN_ID_START,
        speech_token_id_end: int = _DEFAULT_SPEECH_TOKEN_ID_END,
        text_prompt_start_id: int = _DEFAULT_TEXT_PROMPT_START_ID,
        text_prompt_end_id: int = _DEFAULT_TEXT_PROMPT_END_ID,
        # Generation defaults actually used by the reference SDK's
        # `standard.py`/`remote.py` call sites, NOT the checkpoint's own
        # generation_config.json (see docs/Architecture.md Part B.4 -- the
        # checkpoint file's temperature/top_k/top_p are not what upstream
        # inference code uses).
        default_temperature: float = 1.0,
        default_top_k: int = 50,
        default_repetition_penalty: float = 1.2,
        min_new_tokens: int = 50,
        max_context_length: int = 2048,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.use_cache = use_cache

        self.speech_generation_start_id = speech_generation_start_id
        self.speech_generation_end_id = speech_generation_end_id
        self.speech_token_id_start = speech_token_id_start
        self.speech_token_id_end = speech_token_id_end
        self.text_prompt_start_id = text_prompt_start_id
        self.text_prompt_end_id = text_prompt_end_id

        self.default_temperature = default_temperature
        self.default_top_k = default_top_k
        self.default_repetition_penalty = default_repetition_penalty
        self.min_new_tokens = min_new_tokens
        self.max_context_length = max_context_length

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class VieNeuCodecConfig(PretrainedConfig):
    """Codec-stage config for the NeuCodec decoder (see codec.py).

    NeuCodec itself is loaded from its own HF repo (``neuphonic/neucodec`` or
    ``neuphonic/distill-neucodec``) at runtime, not from this checkpoint --
    this config only carries the handful of fields the codec stage needs to
    locate and configure that external codec (docs/Architecture.md Part B.1).
    """

    model_type = "vieneu_codec"

    def __init__(
        self,
        codec_repo_id: str = "neuphonic/neucodec",
        sample_rate: int = 24000,
        input_sample_rate: int = 16000,
        hop_length: int = 480,
        num_codebooks: int = 1,
        codebook_size: int = 65536,
        **kwargs,
    ):
        self.codec_repo_id = codec_repo_id
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.hop_length = hop_length
        # NeuCodec is single-codebook FSQ -- kept explicit (not hardcoded in
        # the decoder) so a future multi-codebook NeuCodec variant does not
        # require touching codec.py's shape assumptions blindly.
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        super().__init__(**kwargs)


class VieNeuConfig(PretrainedConfig):
    """Top-level config for standalone ``AutoConfig.from_pretrained(...)`` use.

    Mirrors the ``FishSpeechConfig`` pattern (docs/Architecture.md Part A.3):
    a top-level config with ``sub_configs`` for each stage, registered with
    ``AutoConfig`` in ``transformers_utils/configs/vieneu.py``. The stage
    pipeline itself does not go through this class -- each stage's engine
    loads ``VieNeuTalkerConfig``/``VieNeuCodecConfig`` directly via
    ``hf_overrides`` in ``pipeline.yaml`` (see registry.py comment).
    """

    model_type = "vieneu"
    sub_configs = {
        "talker_config": VieNeuTalkerConfig,
        "codec_config": VieNeuCodecConfig,
    }

    def __init__(
        self,
        talker_config: dict | VieNeuTalkerConfig | None = None,
        codec_config: dict | VieNeuCodecConfig | None = None,
        **kwargs,
    ):
        if talker_config is None:
            talker_config = {}
        if codec_config is None:
            codec_config = {}

        self.talker_config = (
            talker_config if isinstance(talker_config, VieNeuTalkerConfig) else VieNeuTalkerConfig(**talker_config)
        )
        self.codec_config = (
            codec_config if isinstance(codec_config, VieNeuCodecConfig) else VieNeuCodecConfig(**codec_config)
        )
        super().__init__(**kwargs)

    def get_text_config(self, **kwargs):
        # vLLM expects the text config to expose hidden_size/num_attention_heads.
        return self.talker_config


__all__ = ["VieNeuConfig", "VieNeuTalkerConfig", "VieNeuCodecConfig"]
