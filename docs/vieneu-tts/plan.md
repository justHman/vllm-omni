# PROJECT

Implement native support for **VieNeu-TTS v2** inside **vLLM-Omni** so that it can be served exactly like VoxCPM2 or FishSpeech.

Target command:

```bash
vllm serve pnnbao-ump/VieNeu-TTS-v2 \
    --omni \
    --port 8000
```

The implementation should be production-quality and upstream-friendly.

---

# REFERENCES

## vLLM

https://github.com/vllm-project/vllm

## vLLM-Omni

https://github.com/vllm-project/vllm-omni

## VieNeu-TTS

https://github.com/pnnbao97/VieNeu-TTS

## HuggingFace checkpoint

https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2

Study every file necessary before writing code.

Do NOT guess.

Reuse existing abstractions whenever possible.

---

# GOAL

Add VieNeu as a first-class architecture.

The final user experience should be

```python
OpenAI(
    base_url="http://localhost:8000/v1"
)

client.audio.speech.create(
    model="pnnbao-ump/VieNeu-TTS-v2",
    input="Xin chào"
)
```

without any custom wrapper.

---

# REQUIREMENTS

The implementation must follow the same architecture style used for

* VoxCPM
* VoxCPM2
* FishSpeech
* Ming Flash Omni
* MammothModa2

Do NOT implement an external FastAPI wrapper.

Do NOT bypass vLLM.

Implement proper integration.

---

# TASK 1

Study the architecture of

* voxcpm.py
* voxcpm2.py
* fish_speech.py

Explain

* why each exists
* how registration works
* where model_type comes from
* how stage config is resolved
* how generation works
* how codec is invoked
* how OpenAI endpoint is connected

Produce architecture documentation before coding.

---

# TASK 2

Study the HuggingFace repository.

Determine

* architecture
* tokenizer
* codec
* generation config
* config.json
* tokenizer_config
* special tokens
* sampling
* speech tokens
* speaker embedding
* inference flow

Explain every component.

---

# TASK 3

Determine compatibility.

Answer

Can VieNeu reuse

* VoxCPM2 pipeline

or

* FishSpeech pipeline

or

must have its own implementation.

Provide reasons.

---

# TASK 4

Implement a new architecture.

Expected directory:

vllm_omni/

```
models/

    vieneu/

        __init__.py

        config.py

        tokenizer.py

        processor.py

        codec.py

        generation.py

        registry.py

        stages.py
```

Avoid putting everything into one file.

---

# TASK 5

Implement configuration.

Create

VieNeuConfig

Requirements

* inherit PretrainedConfig
* model_type="vieneu"
* parse config.json
* support loading from HF
* validate required fields

---

# TASK 6

Registration

Register VieNeu into every registry required.

Examples

CONFIG_MAPPING

MODEL_REGISTRY

STAGE_REGISTRY

Any other registry used by vLLM-Omni.

Do NOT use monkey patches.

---

# TASK 7

Tokenizer

Determine whether AutoTokenizer is sufficient.

If not,

implement

VieNeuTokenizer

Requirements

* support text tokens
* support speech tokens
* support BOS/EOS
* support special control tokens
* support batching

---

# TASK 8

Processor

Implement preprocessing.

Pipeline

Text

↓

Normalization

↓

Optional G2P

↓

Prompt construction

↓

Tokenizer

↓

LLM input

Support

speaker

language

emotion (if available)

voice prompt

future extension

---

# TASK 9

Generation

Implement

VieNeuGenerationMixin

Responsibilities

prepare_inputs()

generate()

stream_generate()

speech token collection

EOS detection

---

# TASK 10

Codec

Implement

VieNeuCodec

Responsibilities

decode()

stream_decode()

batch_decode()

GPU inference

Support streaming audio generation.

---

# TASK 11

Stage configuration

Create stage configuration.

Support

Text

↓

LLM

↓

Speech Tokens

↓

Codec

↓

Waveform

Reuse existing scheduler whenever possible.

---

# TASK 12

Streaming

Support

OpenAI streaming

Server Sent Events

incremental audio chunks

low latency

Do not wait until completion.

---

# TASK 13

OpenAI endpoint

Ensure

POST /v1/audio/speech

works exactly like VoxCPM.

No custom endpoints.

---

# TASK 14

Performance

Support

continuous batching

KV cache

prefix cache

GPU scheduling

No regression against existing architectures.

---

# TASK 15

Tests

Unit tests

Config loading

Tokenizer

Codec

Generation

Streaming

Integration tests

HF model loading

OpenAI endpoint

Audio generation

Streaming

---

# TASK 16

Documentation

Produce

Architecture.md

VieNeu.md

Migration.md

Explain

* architecture
* registration
* inference
* streaming
* future extension

---

# CODING RULES

Follow upstream style.

Avoid duplicated code.

Prefer composition over inheritance.

Document public APIs.

Add type hints.

No hardcoded paths.

No magic constants.

Every new module must have tests.

---

# DELIVERABLES

Provide

1.

Architecture analysis

2.

Design document

3.

Implementation

4.

Unit tests

5.

Integration tests

6.

Performance notes

7.

Known limitations

8.

Future improvements

Only mark the project complete if

```bash
vllm serve pnnbao-ump/VieNeu-TTS-v2 --omni
```

starts successfully

and

OpenAI audio endpoint produces valid speech.
