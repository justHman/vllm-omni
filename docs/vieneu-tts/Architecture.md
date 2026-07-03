# VieNeu-TTS-v2 × vllm-omni — Architecture Document

Status: research complete (TASK 1–3), pre-implementation.
Base: fork `justHman/vllm-omni`, branch `feat/vieneu-tts`, from tag `v0.19.0rc1`.

---

## Part A — Existing vllm-omni TTS architecture (TASK 1)

Scope note: the plan referenced `voxcpm.py`/`voxcpm2.py` as primary references. **Those files do not exist at `v0.19.0rc1`** — VoxCPM2 was added later, only on `main`. The rc1 checkout's real TTS architectures are `fish_speech`, `cosyvoice3`, `qwen3_tts` (plus `voxtral_tts`, `mimo_audio`, `omnivoice`). This document uses `fish_speech`, `cosyvoice3`, `qwen3_tts` as the reference set, since those are what the target branch actually contains.

### A.1 Why each architecture exists

- **fish_speech** — dual-AR: a "Slow AR" (`FishSpeechSlowARForConditionalGeneration`, Qwen3 backbone) predicts one semantic codebook token/step; a "Fast AR" (`FishSpeechFastAR`, 4-layer, no KV cache, re-prefills every step) predicts the other 9 residual DAC codebook tokens conditioned on the Slow AR's hidden state. The nested AR-inside-AR is why it needs the GPU-resident `talker_mtp` fast path in the model runner.
- **cosyvoice3** — single `CosyVoice3Model` class, behavior branches on `self.model_stage` (`"cosyvoice3_talker"` vs `"cosyvoice3_code2wav"`), loaded twice with different `model_stage`. Talker is Qwen2-based, flat speech-token stream (no residual codebooks). Code2Wav is CFM diffusion (`n_timesteps`) + HiFi-GAN — categorically different from direct-codec-decode.
- **qwen3_tts** — single-AR talker (Qwen3-based) predicts codec layer-0 tokens; a separate nested `Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM` predicts residual quantizer layers, wired through the same `talker_mtp` fast path as fish_speech. Has an ECAPA-TDNN speaker encoder for zero-shot cloning via `speaker_embedding`. Codec is a residual-VQ tokenizer decoded per-chunk.

### A.2 Registration mechanism

All model classes register in one dict, `_OMNI_MODELS` (`vllm_omni/model_executor/models/registry.py:7-160`), keyed by **architecture class name** (as it appears in `config.json`'s `"architectures"` or in `hf_overrides.architectures`), mapping to `(mod_folder, mod_relname, cls_name)`. `OmniModelRegistry` merges this with vLLM's own `_VLLM_MODELS`, wrapping each entry in `_LazyRegisteredModel` so the class import is deferred until the model loader needs it.

`model_type` (from checkpoint `config.json`, read via `AutoConfig`) is **not** a single-lookup path to an architecture class. Two things must agree:
1. Each pipeline **stage YAML** hardcodes `model_arch: <ClassName>` (or `hf_overrides.architectures: [...]`) per stage — this overrides whatever `architectures` the checkpoint's `config.json` says, because one checkpoint can't list two different top-level architectures for two different stage engines.
2. Each stage runs as an **independent vLLM engine process** with its own `VllmConfig`; that stage's `model_arch`/`hf_overrides.architectures` is what `OmniModelRegistry.resolve_model_cls(...)` (unmodified vLLM logic) uses to import the class.

### A.3 Config resolution

Config classes live in `models/<family>/configuration_*.py`; `AutoConfig.register(...)` side effects live in a **separate** `transformers_utils/configs/<family>.py` module, eagerly imported from `transformers_utils/configs/__init__.py` and from top-level `vllm_omni/__init__.py`.

- **fish_speech** is the only one of the three that does a full `AutoConfig.register("fish_qwen3_omni", FishSpeechConfig)` — it remaps native field names (`dim`, `n_head`, `n_layer`) onto standard Transformers names (`hidden_size`, `num_attention_heads`, `num_hidden_layers`) so vLLM's stock `Qwen3Model` can consume it unmodified.
- **cosyvoice3** and **qwen3_tts** have no `AutoConfig.register` call in rc1 — they rely on `trust_remote_code=True` (set in stage `engine_args`) plus vLLM's own HF-config auto-import, and direct class references (e.g. `self.ctx.get_hf_config(CosyVoice3Config)`).

### A.4 Stage config

A **stage** (`vllm_omni.config.stage_config.StageConfig`) is one node in a DAG of independent vLLM engine processes implementing one model's pipeline end-to-end (e.g. text→tokens is stage 0, tokens→waveform is stage 1). Each stage has `stage_id`, `model_stage` string (keyed by the OpenAI serving layer, see A.7), `stage_type` (`LLM`/`DIFFUSION`), `input_sources`, `worker_type` (`ar`/`generation`), `scheduler_cls`, and its own `yaml_engine_args`.

Two parallel declaration formats exist, both parsed into the same `StageConfig`:
- **New-style `pipeline.yaml`** colocated with the model (only `qwen3_tts` uses this in rc1) — model_type→dir mapping in `StageConfigFactory.PIPELINE_MODELS`.
- **Legacy `stage_configs/<model_type>.yaml`** under `vllm_omni/model_executor/stage_configs/` (`fish_speech`, `cosyvoice3` use this). Resolution reads `hf_config.model_type`, normalizes `-`→`_`, looks up `{model_type}.yaml`. **Known mismatch**: fish_speech's `model_type` is `fish_qwen3_omni` but the file is named `fish_speech_s2_pro.yaml` — every example passes `--stage-configs-path` explicitly rather than relying on auto-resolution.

`custom_process_next_stage_input_func` (a dotted-path string to a function in `stage_input_processors/`) is what connects one stage's raw output to the next stage's input shape — there is no separate "stage registry" beyond this string-import indirection.

### A.5 Generation flow

vLLM's normal decode loop, plus two optional duck-typed lifecycle hooks checked via `getattr`/`hasattr` in `GPUModelRunner` (not a formal ABC):
- **`preprocess(input_ids, input_embeds, **info) -> (input_ids, inputs_embeds, info_update)`** — builds the real `inputs_embeds` (text embed + summed codebook embeds at semantic positions).
- **`postprocess(hidden_states, **_) -> dict`** — stashes last hidden state for the next step.
- **`talker_mtp(...)`** — GPU fast-path running a *nested* residual-codebook AR model in the same CUDA stream as the outer decode step (fish_speech, qwen3_tts only — needed because their codecs have >1 codebook).

EOS is standard vLLM `stop_token_ids` in the stage YAML's `default_sampling_params`, combined with model-side logit masking restricting logits to the valid codebook range plus the stop id. Continuous batching is entirely vLLM's stock `Scheduler`/`Worker` classes — `OmniARScheduler`/`GPUARWorker` for AR stages, `OmniGenerationScheduler`/`GPUGenerationWorker` for codec stages. No custom scheduler logic needed for TTS beyond picking one of these two per stage.

### A.6 Codec invocation

Codec decode is always a **separate pipeline stage** (`worker_type: generation`, `enforce_eager: true`, since dynamic conv/attention shapes in vocoders break CUDA graphs).

- **fish_speech**: `FishSpeechDACDecoder.forward()` lazily loads a DAC codec, bakes weight-norm for inference speed, reshapes flat codebook-major ids to `[num_codebooks, num_frames]`, batches across requests, calls `self._codec.decode(codes_bqf, feature_lengths)` per streaming chunk (~25 frames ≈ 1.16s), trims left-context frames after decode for cross-chunk continuity.
- **qwen3_tts**: `Qwen3TTSCode2Wav.forward()` calls `decoder.chunked_decode()` directly (bypassing the HF tokenizer wrapper) and explicitly enables a codec-specific CUDA graph independent of the stage's own `enforce_eager`.
- **cosyvoice3**: CFM diffusion (`n_timesteps=10`) + HiFi-GAN — categorically slower per chunk than direct decode.

All three set `has_preprocess=False`, `has_postprocess=False`, `requires_raw_input_tokens=True`, `have_multimodal_outputs=True` on the codec-stage class — the shared contract for any codec stage.

### A.7 OpenAI endpoint wiring

Single route, `POST /v1/audio/speech` (`entrypoints/openai/api_server.py`), dispatches to one `OmniOpenAIServingSpeech` instance created at server startup. There is **no adapter registry / `tts_adapters/` directory** in rc1 — model-type detection is by scanning `engine_client.stage_configs` for a stage whose `model_stage` matches a hardcoded set (`_FISH_TTS_MODEL_STAGES`, `_COSYVOICE3_TTS_MODEL_STAGES`, `_QWEN3_TTS_MODEL_STAGES` in `serving_speech.py`), mapped to a short tag via `_detect_tts_model_type()`. Every downstream step — request validation, prompt building, the final `engine_client.generate(...)` call — branches on this tag via `if/elif` chains. Adding a new architecture means adding a new `_VIENEU_TTS_MODEL_STAGES` set, a new `elif self._tts_model_type == "vieneu":` branch, and `_validate_vieneu_request`/`_build_vieneu_prompt` functions, following the fish_speech or cosyvoice3 shape depending on whether the prompt needs raw multimodal audio data or custom tokenized input.

The multimodal output key (`"audio"` vs `"model_outputs"`) is itself model-specific; `_extract_audio_output` checks both.

### A.8 Streaming

Two independent layers:
1. **Pipeline-internal** (talker→codec): `async_chunk: true` in the stage YAML + a `custom_process_next_stage_input_func` (an `*_async_chunk` function) that accumulates decoded frames per request in a `transfer_manager`, decides when enough frames have accumulated (`codec_chunk_frames`/`codec_left_context_frames`/`initial_codec_chunk_frames`), slices a window with left-context overlap, emits the next stage's input dict.
2. **HTTP-facing**: `create_speech()` checks `request.stream`; for `pcm`/`wav` returns a `StreamingResponse` over `_generate_audio_chunks()`, converting each chunk to PCM bytes via `AudioMixin.create_audio()`. A WAV header is emitted once before the first chunk.

### A.9 Reusable abstractions

Reuse directly:
- `vllm_omni.model_executor.models.output_templates.OmniOutput` — the `NamedTuple` every stage `forward()` returns.
- `vllm.model_executor.models.qwen3.Qwen3Model` — if the backbone is Qwen-shaped (it is, for VieNeu — see Part C).
- `vllm_omni.model_executor.stage_input_processors.tts_utils` (`extract_speaker_from_prompt`, etc.) and `chunk_size_utils` (load-adaptive chunking).
- `vllm_omni.config.stage_config.StageConfig`/`ModelPipeline`/`StageConfigFactory` — add a `pipeline.yaml` (new-style, matches qwen3_tts) rather than writing a custom parser.
- `GPUARWorker`/`GPUGenerationWorker` + `OmniARScheduler`/`OmniGenerationScheduler` — select via `worker_type` in the stage YAML.
- The duck-typed flags (`has_preprocess`, `has_postprocess`, `talker_mtp`, `have_multimodal_outputs`, `requires_raw_input_tokens`) — set only what's actually needed.

Must implement from scratch (no shared base covers these anywhere in the codebase):
- `PretrainedConfig` subclass + field remapping (if needed).
- `load_weights()` — every family has bespoke weight-name remapping; no generic TTS weight loader exists.
- The codec/vocoder wrapper module itself (decode + chunking/context-trim logic).
- The `stage_input_processors/<model>.py` chunk-emission function.
- `OmniOpenAIServingSpeech` validation + prompt-building branch — no shared base.

---

## Part B — VieNeu-TTS-v2 architecture (TASK 2)

Source: `github.com/pnnbao97/VieNeu-TTS` (SDK code) + `huggingface.co/pnnbao-ump/VieNeu-TTS-v2` (checkpoint). The repo hosts three generations (v1, v2, v3-Turbo) under one SDK; this section is v2 only, since that's the requested checkpoint.

### B.1 Base architecture

- **LM backbone**: `Qwen3ForCausalLM`, confirmed from `config.json`. Small custom size: 22 layers, hidden_size 768, 12 attention heads / 4 KV heads (GQA), intermediate_size 4096, vocab_size 66938, ~0.3B params, BF16.
- **Codec**: **NeuCodec** (`neuphonic/neucodec`) — FSQ-based, **single codebook** (not RVQ), 50 Hz frame rate, 0.8 kbps, 16-bit codes (65536 possible values — matches the speech-token vocab range below). Encoder input 16 kHz mono, decoder output 24 kHz. Not bundled in the LM checkpoint — a separate HF repo/package loaded independently. Decode-only in most inference paths; encode-only for voice cloning (reference audio → reference tokens).
- No known upstream base checkpoint for the Qwen3 weights beyond this — trained from scratch by the author on ~10k hours of bilingual VI/EN data. "VieNeu" = author's brand (Vie = Vietnamese, Neu = built on Neuphonic's NeuCodec), not a fine-tune of a separate "NeuTTS" product.

**v3-Turbo is architecturally different** (different codec — `MOSS-Audio-Tokenizer-Nano` — 48kHz output, 10 named presets with `reserved_id` speaker tokens) and is out of scope; do not conflate its parameters (`max_new_frames`, 10 voices) with v2.

### B.2 config.json (retrieved in full)

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 768,
  "num_hidden_layers": 22,
  "num_attention_heads": 12,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "intermediate_size": 4096,
  "vocab_size": 66938,
  "max_position_embeddings": 4096,
  "eos_token_id": 375,
  "pad_token_id": 375,
  "tie_word_embeddings": true,
  "rope_theta": 1000000,
  "rms_norm_eps": 1e-06
}
```

**Critical fact**: `model_type` is the vanilla `"qwen3"`, not a custom TTS model_type. No nested codec config is embedded — the codec is a fully separate model, loaded independently by the SDK.

### B.3 Tokenizer

`Qwen2Tokenizer` (standard AutoTokenizer subclass), vocab_size 66938. `eos_token`/`pad_token`/`unk_token` are all `<|endoftext|>` (id 375, the only token marked `"special": true`). No separate BOS.

Extended-vocabulary layout (from `added_tokens_decoder`):

| ID range | Token(s) | Purpose |
|---|---|---|
| 375 | `<\|endoftext\|>` | tokenizer-default EOS/PAD/UNK |
| 376 | `<\|TEXT_REPLACE\|>` | chat-template placeholder |
| 377 / 378 | `<\|TEXT_PROMPT_START\|>` / `<\|TEXT_PROMPT_END\|>` | text-region delimiters |
| 379 | `<\|SPEECH_REPLACE\|>` | chat-template placeholder |
| 380 | `<\|SPEECH_GENERATION_START\|>` | speech-region start |
| 381 | `<\|SPEECH_GENERATION_END\|>` | **the actual inference-time stop token** (not the tokenizer's default EOS 375 — see B.5) |
| 382–65917 | `<\|speech_0\|>` … `<\|speech_65535\|>` | 65,536 NeuCodec codebook tokens |
| 65918–66917 | `<\|speaker_0\|>` … `<\|speaker_999\|>` | 1,000 speaker tokens — **not used anywhere in the v2 SDK code path**; status unclear, possibly vestigial |
| 66918–66937 | `<\|emotion_0\|>` … `<\|emotion_19\|>` | 20 emotion tokens; only emotion_0–3 (natural/chuckle/sigh/clear-throat) are used by code paths confirmed for v2 |

None of the added tokens except 375 are flagged `"special": true` — they will appear in raw `tokenizer.decode()` output; the SDK extracts speech tokens via regex (`<\|speech_(\d+)\|>`) post-decode rather than relying on tokenizer-level skip behavior.

### B.4 Generation config — checkpoint file vs. actual SDK defaults

`generation_config.json` on the Hub (`temperature=0.7, top_k=20, top_p=0.8, eos_token_id=[375]`) is **not what the SDK actually uses**. The SDK's own call sites override at runtime:

| Backend | temperature | top_k | top_p | repetition_penalty | min_new_tokens |
|---|---|---|---|---|---|
| Torch (`standard.py`) | 1.0 | 50 | — | — | 50 |
| Remote/LMDeploy (`remote.py`) | 1.0 | 50 | — | 1.2 | server-side |
| Fast/LMDeploy (`fast.py`) | 1.0 | 50 | 0.95 | 1.2 | 40 |

Stop condition is always `<|SPEECH_GENERATION_END|>` (id 381), passed explicitly as `eos_token_id`/`stop=[...]` per backend — **not** the checkpoint's default `eos_token_id=375`. No `max_new_tokens`/`max_new_frames` field exists for v2; generation is bounded by `max_length=2048` (total sequence cap) combined with the stop token.

**Integration must replicate the SDK's call-site defaults, not the checkpoint's `generation_config.json` values.**

### B.5 Speech tokens — sequence layout

Text-first, then speech (not interleaved token-by-token):

```
<|TEXT_PROMPT_START|>{emotion_tag}{ref_text_phonemes} {input_text_phonemes}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{ref_speech_tokens}[...generated speech tokens...]<|SPEECH_GENERATION_END|>
```

v2 does **not** use the chat-style wrapper that exists for v1 (`use_chat_format` is false whenever the backbone repo isn't literally `pnnbao-ump/VieNeu-TTS`).

### B.6 Voice cloning mechanism

**No learned speaker-embedding vector.** Voice identity is carried entirely by reference-audio-as-token-prefix continuation:
1. Reference wav → 16kHz mono → NeuCodec encoder → `ref_codes` (integer sequence).
2. `ref_codes` placed as literal `<|speech_i|>` tokens right after `<|SPEECH_GENERATION_START|>`.
3. A matching **reference transcript** (`ref_text`) is required — phonemized and placed in the text region before the input text. **Voice cloning is not zero-shot from audio alone; a transcript string is mandatory** (either user-supplied or bundled in a preset).
4. Named presets (`voices.json`) store precomputed `{codes, text, description}` per voice — **not** audio files, not embedding vectors.

**Correction to the plan's premise**: the plan/earlier Gradio trace describes **10 named voices** with a `max_new_frames` slider — that parameter set matches **v3-Turbo**, not v2. The v2 checkpoint's `voices.json` has exactly **7 presets** (`Binh`, `Tuyen`, `Vinh`, `Doan`, `Ly`, `Sơn`, `Ngoc`; default `Ly`), no `max_new_frames` parameter, and the voices are licensed **CC BY-NC 4.0** (non-commercial) — stricter than the model weights' own Apache-2.0 license. If the integration target is genuinely `pnnbao-ump/VieNeu-TTS-v2` (Qwen3 + NeuCodec, per its `config.json`), plan around 7 presets / no `max_new_frames`, and flag this discrepancy rather than silently importing v3-Turbo's parameter shape.

### B.7 Inference flow (v2)

1. Text normalization + chunking — external `sea-g2p` package normalizes text (numbers/dates/currency/punctuation), splits into chunks bounded by `max_chars` (default 256), sentence-boundary-aware.
2. G2P — `sea_g2p.G2P.phonemize_batch(...)`, external dependency, bilingual VI/EN.
3. Reference resolution — encode reference audio to `ref_codes`, or use a preset's precomputed codes.
4. Prompt construction — per B.5 layout.
5. Tokenization — phoneme text region via standard `tokenizer.encode`; each `<|speech_i|>` is a literal vocab entry, concatenated directly.
6. LLM generation — standard HF `generate()` (KV cache on, sampling on, `min_new_tokens=50`, stop at id 381 or `max_length=2048`).
7. Speech-token extraction — regex `<\|speech_(\d+)\|>` over the decoded output tail (control tokens are simply not matched, not filtered by the tokenizer).
8. Codec decode — reshape extracted codes to `[1, 1, num_frames]`, `codec.decode_code(codes)` → raw waveform.
9. Optional watermarking (`perth` package, silently skipped if absent).
10. Multi-chunk joining — silence padding or crossfade between independently-inferred chunks.
11. Streaming — `infer_stream()`: windowed incremental decode of the *same* generation stream (accumulate new tokens → decode a window with lookback/lookforward/overlap → `_linear_overlap_add` → yield only new samples). Genuinely incremental, not generate-then-chunk.

### B.8 License

Model weights + SDK code: Apache-2.0. Bundled preset voices (`voices.json`): **CC BY-NC 4.0, non-commercial** — a materially stricter, separate license from the weights. NeuCodec: Apache-2.0.

---

## Part C — Compatibility decision (TASK 3)

**Question**: can VieNeu-TTS-v2 reuse the fish_speech pipeline, the cosyvoice3 pipeline, or does it need its own implementation?

**Answer: VieNeu needs its own implementation, but it is structurally the *simplest* of the four vllm-omni TTS architectures — simpler than qwen3_tts, not just "different."**

Reasoning, mapped against each existing pipeline:

1. **Not fish_speech-compatible.** fish_speech's entire raison d'être is a dual-AR split (Slow AR + Fast AR) to predict 10 DAC codebooks per frame, requiring `talker_mtp`, codebook-embedding summation in `preprocess()`, and codebook-major reshaping in the decoder. NeuCodec is a **single FSQ codebook** — there is nothing to sum, nothing to predict via a nested fast AR, and no multi-codebook reshape in the decoder. Reusing fish_speech's classes would mean stripping out the one thing they're built for.

2. **Not cosyvoice3-compatible.** cosyvoice3's Code2Wav stage is a CFM diffusion process + HiFi-GAN vocoder — VieNeu's NeuCodec is a direct decode-only neural codec (`codec.decode_code(codes)`, no diffusion timesteps). Different stage_type (`DIFFUSION` vs `LLM`/generation-worker), different compute profile, different config surface (`n_timesteps` doesn't exist for VieNeu).

3. **Closer to qwen3_tts in shape, but simpler.** qwen3_tts is the best structural analog: single-AR Qwen-based talker, generation config with a codec-specific stop token, decode-only codec stage. But qwen3_tts's talker *also* needs `talker_mtp` because its codec is residual-VQ (multiple quantizer layers per frame, hence the nested `Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM`). VieNeu's NeuCodec has **one flat token stream** — there is no residual layer to predict, so **no nested code-predictor AR is needed at all**. The talker stage can be closer to a stock vLLM Qwen3 causal-LM decode loop than any of the three existing TTS architectures.

4. **A genuinely novel wrinkle none of the three existing architectures have**: VieNeu's checkpoint `config.json` declares `model_type: "qwen3"` and `architectures: ["Qwen3ForCausalLM"]` — i.e., **on disk it is not distinguishable from a generic Qwen3 LM**. fish_speech/cosyvoice3/qwen3_tts all ship a custom `model_type` in their checkpoint's `config.json`, so vllm-omni's model_type→pipeline resolution "just works" from the checkpoint alone. VieNeu requires an explicit `hf_overrides` (in the stage `pipeline.yaml`, following the same mechanism qwen3_tts already uses for `architectures` override) to present a distinguishing `model_type`/`architectures` pair to vllm-omni's registry **without modifying the upstream checkpoint's `config.json`** — this satisfies the spec's "FAIL: manual config edits required" constraint while still letting the pipeline resolve correctly.

**Conclusion**: implement a new architecture family `vieneu` under `vllm_omni/model_executor/models/vieneu/`, with:
- **Stage 0 (talker)**: a thin subclass wrapping vLLM's stock `Qwen3Model`/`Qwen3ForCausalLM` machinery (reuse, don't reimplement the transformer stack) — the only real additions are (a) EOS/stop-token override to id 381 (`<|SPEECH_GENERATION_END|>`) instead of the checkpoint's default 375, (b) `compute_logits()` masking to the valid speech-token id range (382–65917) plus the stop id, matching the fish_speech/qwen3_tts pattern of restricting the sampled vocabulary. **No `preprocess`/`postprocess`/`talker_mtp` hooks are needed** — a single codebook means standard embedding lookup and standard sampling already do the right thing.
- **Stage 1 (codec decoder)**: a new `VieNeuCodecDecoder` wrapping NeuCodec's `decode_code()`, structurally mirroring `FishSpeechDACDecoder`'s batching/GPU-placement pattern but *without* its multi-codebook reshape (single codebook = simpler `[1, 1, num_frames]` reshape, matching what the reference SDK already does).
- **Registration**: `hf_overrides={"model_type": "vieneu_talker", "architectures": ["VieNeuTalkerForConditionalGeneration"]}` in `pipeline.yaml` (qwen3_tts precedent), `AutoConfig.register` for a `VieNeuConfig(PretrainedConfig)` with `model_type="vieneu"` for standalone `AutoConfig.from_pretrained` usability (fish_speech precedent), plus the `_OMNI_MODELS` registry entries for both stage classes.
- **OpenAI wiring**: new `_VIENEU_TTS_MODEL_STAGES` set, `elif self._tts_model_type == "vieneu":` branch, `_validate_vieneu_request`/`_build_vieneu_prompt` in `serving_speech.py`, following fish_speech's tokenized-prompt shape (not cosyvoice3's raw-multimodal-data shape, since VieNeu's "reference audio" input is converted to token-prefix text before ever reaching the LM, not passed as a `MultiModalDataDict`).
- **Processor**: must call the external `sea-g2p` package for normalization/phonemization (same as the reference SDK) — this is a new runtime dependency, not something reusable from existing vllm-omni code.
- **Presets**: ship the 7 v2 presets from `voices.json`, respecting the CC BY-NC 4.0 license note distinct from the Apache-2.0 model weights; do not import v3-Turbo's 10-voice/`max_new_frames` parameter shape.

This conclusion directly changes two items in the original plan/spec that should be corrected before implementation:
- **TASK 7 (tokenizer)**: `AutoTokenizer` (`Qwen2Tokenizer`) is sufficient as-is — no custom `VieNeuTokenizer` class is needed. The only "special" handling (speech-token regex extraction, control-token IDs) belongs in the **processor**, not a tokenizer subclass.
- **FR-6 / voices**: correct the spec's implicit 10-voice assumption to the verified 7 presets for v2; `max_new_frames` as a user-facing param does not apply to v2 (bounded instead by `max_length=2048` + stop token).

---

## Next steps (not yet done)

TASK 4–16 (directory scaffold, `VieNeuConfig`, registry wiring, processor, generation, codec, stage config, streaming, OpenAI endpoint, tests, docs) are unstarted. This document is the required pre-coding deliverable for TASK 1–3.
