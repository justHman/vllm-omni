# Technical Specification

## Objective

Integrate VieNeu-TTS into vLLM-Omni as a native architecture.

---

## Functional Requirements

### FR-1

HF model loading

Input

```
pnnbao-ump/VieNeu-TTS-v2
```

Output

Loaded successfully.

---

### FR-2

AutoConfig

Support

```
model_type = vieneu
```

---

### FR-3

Tokenizer

Support

* batching
* BOS
* EOS
* special tokens
* speech tokens

---

### FR-4

Generation

Return

Speech Tokens

instead of text.

---

### FR-5

Codec

Convert speech tokens

↓

PCM waveform

↓

WAV

---

### FR-6

Streaming

Audio chunks

<250 ms latency target.

---

### FR-7

OpenAI API

Support

```
POST /v1/audio/speech
```

```
stream=true
```

---

## Non-functional Requirements

Startup

<30 s

Memory

No more than existing model footprint.

Thread-safe

Yes.

Continuous batching

Required.

KV cache

Required.

Prefix cache

Required.

---

## Directory Layout

```
vllm_omni/

    models/

        vieneu/

            config.py

            processor.py

            tokenizer.py

            codec.py

            generation.py

            registry.py

            stages.py

            tests/

configs/

    vieneu.yaml

docs/

    vieneu.md

tests/

    test_vieneu.py
```

---

## Acceptance Criteria

PASS

* HF checkpoint loads.
* Model registers automatically.
* `vllm serve ... --omni` starts.
* `/v1/models` lists VieNeu.
* `/v1/audio/speech` returns valid WAV.
* Streaming works.
* Existing architectures remain unaffected.

FAIL

* Monkey patches required.
* Separate FastAPI wrapper required.
* Manual config edits required.
* Existing models break.
