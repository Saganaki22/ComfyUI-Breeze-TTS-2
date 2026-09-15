# ComfyUI-Breeze-TTS-2

**English** | [简体中文](README_ZH.md)

<img width="1483" height="1170" alt="Screenshot 2026-08-26 160830" src="https://github.com/user-attachments/assets/95f2f346-472e-45a5-bd3e-625007c6cc39" />



https://github.com/user-attachments/assets/d4226032-3337-450b-83c6-ab9c732172d6




[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) nodes for ComfyUI: **voice clone**,
**voice design**, and **voice direction** with bilingual (EN/ZH) speech, inline vocal events,
INT8 ConvRot quantized builds, and full ComfyUI / AIMDO dynamic-VRAM integration.

## Nodes (category: `Breeze TTS 2`)

| Node | Purpose |
| --- | --- |
| **Breeze TTS 2 Load Model** | Loads the checkpoint with `dtype` (auto/bf16/fp32), `device` (auto/cuda/cpu), `attention` (auto/eager/sdpa/flash_attention/sageattention), `decode_mode` (eager/cuda_graphs), and `download_if_missing`. |
| **Breeze TTS 2 Voice Clone** | Clones a speaker from clean reference audio + its exact transcript (CFG 1.0). |
| **Breeze TTS 2 Voice Design** | Creates a voice from a natural-language description, no reference audio (CFG 4). |
| **Breeze TTS 2 Voice Direction** | Clones a reference voice and steers tone, emotion, pace, and delivery with an instruction (CFG 4). `stitch_reference` can play the original reference clip before or after the generated speech in the output. |
| **Breeze TTS 2 Whisper Transcribe** | Whisper helper that produces the exact transcript cloning needs. |
| **Breeze TTS 2 Speaker** | One named cast member for Multi-Speaker. Pick a reference clip from the input folder (browse/upload dropdown, or wire audio in) to clone it, or select `none` and write an instruction to design the voice. An empty transcript is auto-transcribed with Whisper and shown on the node. |
| **Breeze TTS 2 Multi-Speaker** | Generates a whole dialogue script — plain `Name:` lines or pasted JSON — with up to 8 wired speakers, freely mixing cloned and designed voices. |

Vocal events work inline in the text: `(laugh)`, `(cough)`, `(clears throat)`, `(sigh)` in English;
`[笑]`, `[咳嗽]`, `[清嗓子]`, `[叹气]` in Chinese. Match the instruction language to the text language
for design/direction. Output is 24 kHz mono.

## Multi-speaker dialogue



https://github.com/user-attachments/assets/bf124113-f9ef-4a5d-a501-5935d8fd7b60



Wire up to 8 **Speaker** nodes into the **Multi-Speaker** node's `speaker_1…8` inputs, then write
(or paste) the script. Speaker matching is forgiving about case, spaces, and punctuation
(`Ali G` matches a speaker named `alig`), and every speaker name is validated before any
generation starts, so a bad script fails instantly instead of after burning GPU time.

**Option 1 — plain text.** One `Name: line` per line. A line without a name continues the
previous speaker; `[Name]:` and markdown `**Name:**` also work:

```
Ada: (sigh) Who steered my ship into the harbor wall this time?
Bob: Technically, captain, the harbor steered into us.
Ada: That is not how harbors work, Bob.
Bob: If I may, the tide charts were very clear about it.
```

**Option 2 — pasted JSON.** Ideal for LLM-written scripts. If the text starts with `[` or `{`
it is parsed as JSON:

```json
[
  {"speaker": "Ada", "text": "(sigh) Who steered my ship into the harbor wall this time?"},
  {"speaker": "Bob", "text": "Technically, captain, the harbor steered into us."},
  {"speaker": "Ada", "text": "That is not how harbors work, Bob."}
]
```

Also accepted: a single `{"speaker": ..., "text": ...}` object, a wrapper like
`{"script": [...]}`, and the key aliases `speaker`/`name`/`character`/`role` and
`text`/`line`/`content`/`message`. Inline vocal events work identically in both formats.
The `text` widget can be converted to an input (right-click) to wire a string from any
text-generating node.

Before inference the console prints a summary —

```
[BreezeTTS2] Multi-speaker script: 4 speaker(s), 16 turn(s), ~75s of speech total | cast: Ada (clone), Bob (clone), Carol (design), Dave (design)
```

— then one progress bar per turn (`[3/16] Bob (~6s)`). Reference clips are encoded once per
speaker, each speaker keeps a stable seed offset across turns (so designed voices stay
consistent), and `pause_between_speakers` controls the gap. Reference clips: keep them under
~20 s (hard max 60 s). To get an LLM to write scripts in this format, hand it
[SKILL.md](SKILL.md).

<img width="776" height="1153" alt="Screenshot 2026-08-28 035631" src="https://github.com/user-attachments/assets/be2ad864-bb9d-42b2-a474-ad22315a4dad" />


## Checkpoints

The loader downloads into `ComfyUI/models/breezetts2/` when `download_if_missing` is enabled:

| Choice | File | Notes |
| --- | --- | --- |
| int8 hybrid **(default)** | `Breeze-TTS-2-int8-hybrid.safetensors` | Backbone + text encoder INT8 ConvRot, depth decoder bf16. bf16-level speed at −27 % VRAM. |
| bf16 combined | `Breeze-TTS-2-bf16.safetensors` | Same weights merged into one file (bit-exact). Fastest. |
| INT8 all linears | `Breeze-TTS-2-int8-convrot.safetensors` | Smallest build; slower in the decode loop (see below). |
| INT8 text encoder only | `Breeze-TTS-2-int8-text-encoder.safetensors` | Decode path fully bf16; smallest quality-irrelevant VRAM cut. |

All mirrors live in [drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui).

## Model Files

Weights are stored per source repo under `ComfyUI/models/breezetts2/` (primary folder first; symlinks are searched too). Every models tree mapped through `extra_model_paths.yaml` is also searched, including sections that only list standard folders like `checkpoints:` — a `breezetts2/` folder inside any mapped tree is picked up, so one fixed location can serve every ComfyUI install. Mapped folders are read-only to the node: new downloads always land in the running install's own models folder. An explicit `breezetts2:` entry in the yaml works as well. Missing files download automatically when `download_if_missing` is on, otherwise the error names the expected folder.

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 breezetts2/
        └── 📂 drbaph_Breeze-TTS-2-comfyui/      (all builds, default source)
            ├── Breeze-TTS-2-bf16.safetensors              (6.49 GiB, official shards merged bit-exact)
            ├── Breeze-TTS-2-int8-hybrid.safetensors       (4.53 GiB, recommended int8 build)
            ├── Breeze-TTS-2-int8-convrot.safetensors      (4.22 GiB, all 462 linears quantized)
            ├── Breeze-TTS-2-int8-text-encoder.safetensors (5.84 GiB, decode path bf16)
            ├── config.json
            ├── generation_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── special_tokens_map.json
            └── 📂 audio_tokenizer/                        (Qwen3-TTS 12 Hz codec, shared)
                ├── model.safetensors                      (0.64 GiB)
                ├── config.json
                ├── configuration.json
                └── preprocessor_config.json
```

Only the selected weights file is downloaded; every build shares the same `config.json`, tokenizer files, and `audio_tokenizer/` codec.

| Component | bf16 build | int8 builds |
| --- | --- | --- |
| Model weights | 6.49 GiB | 4.22–5.84 GiB |
| `audio_tokenizer` codec | 0.64 GiB | unchanged |
| Tokenizer + configs | ~35 MB | unchanged |

Expect roughly **5.3–7.5 GiB VRAM** depending on build and dtype; AIMDO DynamicVRAM pages castable weights to keep live VRAM pressure low alongside other models.

## Benchmarks (RTX 5090, ComfyUI loader, sdpa)

| Build | avg RTF | Peak VRAM | Whisper WER vs prompt |
| --- | --- | --- | --- |
| bf16 | 5.68 | 7.51 GiB | baseline |
| int8-convrot (all linears) | 9.08 | 5.21 GiB | matches bf16 |
| **int8-hybrid (depth decoder bf16)** | **5.64** | **5.53 GiB** | matches bf16 |
| int8-text-encoder only | 5.34 | 6.82 GiB | matches bf16 |

Why full INT8 regresses: the depth decoder runs 15 skinny single-token GEMMs per 12.5 Hz frame
(≈1260 linear calls/frame). At M=1–2 the int8 per-call overhead (~0.1 ms quantize + dispatch +
allocations) dwarfs the GEMM savings, so those layers lose to plain bf16 `F.linear` (~0.03 ms).
The backbone contributes only ~196 calls/frame and the text encoder runs once at prefill, so
quantizing them is nearly free — that is exactly what the hybrid build does. Full measurement
data is in `benchmark_report.json` on the mirror repo.

## Requirements

- ComfyUI with **Transformers 4.57 or 5.3+**. On 5.x the text encoder uses the native
  `T5Gemma2TextEncoder` wired into flash attention through a custom-registered FA2 path
  (upstream blocks it over softcapping concerns this checkpoint does not have); on 4.x it
  uses the vendored upstream compat implementation, which also runs flash attention.
- torch/torchaudio from your ComfyUI install. `flash_attn` and `sageattention` are optional.
- `comfy-kitchen` ships with current ComfyUI and is only imported when an INT8 checkpoint is loaded.

## Model architecture (what actually runs)

- **Text encoder**: T5Gemma2 (26 layers, 1152 hidden) + linear projection into the backbone space.
- **Backbone**: Qwen3-style decoder (28 layers, 2048 hidden) predicting the first codebook token per
  12.5 Hz frame, with an audio-token embedding summed over the 16 codebooks.
- **Depth decoder**: 12-layer decoder generating codebooks 1–15 per frame with its own CFG.
- **Audio tokenizer**: vendored Qwen3-TTS 12 Hz codec (Mimi-style encoder + streaming decoder),
  24 kHz, 16 codebooks. Reference audio is encoded with the same codec.
- The in-checkpoint `codec_model` (a Mimi copy used only by the upstream training forward) is not
  built at inference.

The eager generation loop reimplements the official `FastBreezeStreamingRuntime` with all fast
stages disabled (DynamicCache instead of CUDA graphs) so it stays compatible with AIMDO dynamic
VRAM paging and INT8 weight casting. With `decode_mode=cuda_graphs` the depth decode steps are
captured into CUDA graphs instead — the captured decode runs **7–9x faster** than the eager
AIMDO-paged path (~3–4x faster end-to-end generation); that mode keeps the model weights
VRAM-resident rather than AIMDO-paged. The official CUDA-graph fast path is intentionally not ported.

## License

Node-pack code: Apache-2.0 (see `LICENSE`). Vendored model code keeps its upstream licenses
(Apache-2.0 from breezeblue-ai/breeze-tts and Qwen/Qwen3-TTS).

**Model weights, derivative checkpoints, and self-hosted outputs are governed by the BreezeBlue
Research and Non-Commercial License** — see [MODEL_LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
Commercial use requires written authorization from RESONIA, INC.
