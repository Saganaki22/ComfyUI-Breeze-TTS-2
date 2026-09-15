# ComfyUI-Breeze-TTS-2

[English](README.md) | **简体中文**

<img width="1483" height="1170" alt="Screenshot 2026-08-26 160830" src="https://github.com/user-attachments/assets/95f2f346-472e-45a5-bd3e-625007c6cc39" />

https://github.com/user-attachments/assets/d4226032-3337-450b-83c6-ab9c732172d6

适用于 ComfyUI 的 [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) 节点：**语音克隆**、
**声音设计** 与 **声音风格指令**，支持中英双语语音、内联语气事件、
INT8 ConvRot 量化构建，并完整集成 ComfyUI / AIMDO 动态显存管理。

## 节点（分类：`Breeze TTS 2`）

| 节点 | 用途 |
| --- | --- |
| **Breeze TTS 2 Load Model** | 加载检查点，支持 `dtype`（auto/bf16/fp32）、`device`（auto/cuda/cpu）、`attention`（auto/eager/sdpa/flash_attention/sageattention）、`decode_mode`（eager/cuda_graphs）以及 `download_if_missing`。 |
| **Breeze TTS 2 Voice Clone** | 通过干净的参考音频 + 精确转写文本克隆说话人（CFG 1.0）。 |
| **Breeze TTS 2 Voice Design** | 通过自然语言描述创建声音，无需参考音频（CFG 4）。 |
| **Breeze TTS 2 Voice Direction** | 克隆参考声音，并用指令控制语气、情绪、语速和表达方式（CFG 4）。`stitch_reference` 可将原始参考片段拼接到生成语音之前或之后输出。 |
| **Breeze TTS 2 Whisper Transcribe** | Whisper 辅助节点，生成克隆所需的精确转写文本。 |
| **Breeze TTS 2 Speaker** | 多说话人节点中的一个具名角色。从 input 目录选择参考音频（下拉框支持浏览/上传，也可连线音频输入）进行克隆；或选择 `none` 并填写指令来设计声音。转写文本留空时会用 Whisper 自动转写，并显示在节点上。 |
| **Breeze TTS 2 Multi-Speaker** | 一次性生成整段对话脚本 —— 支持纯文本 `名字:` 行或直接粘贴 JSON —— 最多接入 8 个说话人，可自由混合克隆声音与设计声音。 |

语气事件可直接写在文本中：英文用 `(laugh)`、`(cough)`、`(clears throat)`、`(sigh)`；
中文用 `[笑]`、`[咳嗽]`、`[清嗓子]`、`[叹气]`。设计/指令节点的指令语言应与文本语言
一致。输出为 24 kHz 单声道。

## 多说话人对话

将最多 8 个 **Speaker** 节点接入 **Multi-Speaker** 节点的 `speaker_1…8` 输入，然后编写
（或粘贴）脚本。说话人名字匹配对大小写、空格和标点宽容（`Ali G` 可匹配名为 `alig` 的说话人），
且所有名字都会在生成开始前校验 —— 脚本写错会立即报错，不会白白消耗显卡时间。

**方式一 —— 纯文本。** 每行一个 `名字: 台词`。没有名字的行会并入上一位说话人；
`[名字]:` 和 markdown 的 `**名字:**` 写法同样支持：

```
Ada: (sigh) 这次又是谁把我的船撞上码头了？
Bob: 严格来说，船长，是码头撞上了我们。
Ada: 码头可不是这样运作的，Bob。
Bob: 恕我直言，潮汐表上写得清清楚楚。
```

**方式二 —— 粘贴 JSON。** 适合 LLM 生成的脚本。只要文本以 `[` 或 `{` 开头，
就会按 JSON 解析：

```json
[
  {"speaker": "Ada", "text": "(sigh) 这次又是谁把我的船撞上码头了？"},
  {"speaker": "Bob", "text": "严格来说，船长，是码头撞上了我们。"},
  {"speaker": "Ada", "text": "码头可不是这样运作的，Bob。"}
]
```

同时还接受：单个 `{"speaker": ..., "text": ...}` 对象、`{"script": [...]}` 这类包装，
以及键名别名 `speaker`/`name`/`character`/`role` 和 `text`/`line`/`content`/`message`。
内联语气事件在两种格式中完全相同。`text` 控件可转换为输入（右键菜单），
从而接入任意文本生成节点。

推理开始前，控制台会先打印摘要 ——

```
[BreezeTTS2] Multi-speaker script: 4 speaker(s), 16 turn(s), ~75s of speech total | cast: Ada (clone), Bob (clone), Carol (design), Dave (design)
```

—— 然后每个回合显示一条进度条（`[3/16] Bob (~6s)`）。参考音频每位说话人只编码一次；
每位说话人在所有回合中使用固定的种子偏移（设计的声音因此保持一致）；
`pause_between_speakers` 控制回合间的停顿。参考音频建议不超过约 20 秒（硬性上限 60 秒）。
想让 LLM 按此格式编写脚本，把 [SKILL.md](SKILL.md) 交给它即可。

## 检查点

启用 `download_if_missing` 时，加载器会下载到 `ComfyUI/models/breezetts2/`：

| 选项 | 文件 | 说明 |
| --- | --- | --- |
| int8 hybrid **（默认）** | `Breeze-TTS-2-int8-hybrid.safetensors` | 主干 + 文本编码器 INT8 ConvRot，深度解码器保持 bf16。速度持平 bf16，显存降低 27%。 |
| bf16 combined | `Breeze-TTS-2-bf16.safetensors` | 官方分片权重合并为单文件（逐位一致）。速度最快。 |
| INT8 all linears | `Breeze-TTS-2-int8-convrot.safetensors` | 最小构建；解码循环更慢（见下文）。 |
| INT8 text encoder only | `Breeze-TTS-2-int8-text-encoder.safetensors` | 解码路径完全 bf16；在不影响质量的前提下最小幅度削减显存。 |

所有镜像均位于 [drbaph/Breeze-TTS-2-comfyui](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui)。

## 模型文件

权重按来源仓库存放在 `ComfyUI/models/breezetts2/` 下（优先主目录；符号链接同样会被搜索）。通过 `extra_model_paths.yaml` 映射的模型目录树也都会被搜索——包括只映射了 `checkpoints:` 等标准文件夹的配置段：只要映射目录里存在 `breezetts2/` 文件夹即可被识别，这样一个固定位置就能同时服务多个 ComfyUI 安装。映射目录对节点是只读的：新下载始终落在当前运行的 ComfyUI 自己的 models 目录。在 yaml 中显式写 `breezetts2:` 条目同样有效。开启 `download_if_missing` 时缺失文件会自动下载，否则报错信息会指出预期目录。

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 breezetts2/
        └── 📂 drbaph_Breeze-TTS-2-comfyui/      （全部构建，默认来源）
            ├── Breeze-TTS-2-bf16.safetensors              （6.49 GiB，官方分片逐位一致合并）
            ├── Breeze-TTS-2-int8-hybrid.safetensors       （4.53 GiB，推荐的 int8 构建）
            ├── Breeze-TTS-2-int8-convrot.safetensors      （4.22 GiB，全部 462 个线性层已量化）
            ├── Breeze-TTS-2-int8-text-encoder.safetensors （5.84 GiB，解码路径 bf16）
            ├── config.json
            ├── generation_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── special_tokens_map.json
            └── 📂 audio_tokenizer/                        （Qwen3-TTS 12 Hz 编解码器，共享）
                ├── model.safetensors                      （0.64 GiB）
                ├── config.json
                ├── configuration.json
                └── preprocessor_config.json
```

仅下载所选权重文件；所有构建共用同一份 `config.json`、分词器文件与 `audio_tokenizer/` 编解码器。

| 组件 | bf16 构建 | int8 构建 |
| --- | --- | --- |
| 模型权重 | 6.49 GiB | 4.22–5.84 GiB |
| `audio_tokenizer` 编解码器 | 0.64 GiB | 不变 |
| 分词器 + 配置文件 | ~35 MB | 不变 |

预计显存占用约 **5.3–7.5 GiB**（取决于构建与 dtype）；AIMDO DynamicVRAM 会对可转换的权重进行分页，以便在与其他模型并存时保持较低的实时显存压力。

## 基准测试（RTX 5090，ComfyUI 加载器，sdpa）

| 构建 | 平均 RTF | 峰值显存 | Whisper WER（对比提示文本） |
| --- | --- | --- | --- |
| bf16 | 5.68 | 7.51 GiB | 基准 |
| int8-convrot（全部线性层） | 9.08 | 5.21 GiB | 与 bf16 一致 |
| **int8-hybrid（深度解码器 bf16）** | **5.64** | **5.53 GiB** | 与 bf16 一致 |
| int8-text-encoder only | 5.34 | 6.82 GiB | 与 bf16 一致 |

全量 INT8 变慢的原因：深度解码器在每个 12.5 Hz 帧上执行 15 次窄小的单 token GEMM
（≈每帧 1260 次线性层调用）。当 M=1–2 时，int8 的每次调用开销（约 0.1 ms 的量化 + 调度 +
内存分配）远超 GEMM 节省的时间，因此这些层反而输给纯 bf16 `F.linear`（约 0.03 ms）。
主干每帧仅贡献约 196 次调用，文本编码器只在 prefill 运行一次，量化它们几乎无成本
——这正是 hybrid 构建的做法。完整测量数据见镜像仓库中的 `benchmark_report.json`。

## 运行要求

- 搭载 **Transformers 4.57 或 5.3+** 的 ComfyUI。5.x 下文本编码器使用原生
  `T5Gemma2TextEncoder`，并通过自定义注册的 FA2 路径运行 flash attention
  （上游因 softcapping 顾虑禁用了它，但本检查点并不涉及）；4.x 下使用随附的上游兼容实现，
  同样可以运行 flash attention。
- torch/torchaudio 来自你的 ComfyUI 安装。`flash_attn` 与 `sageattention` 为可选项。
- `comfy-kitchen` 随当前 ComfyUI 附带，仅在加载 INT8 检查点时才会导入。

## 模型架构（实际运行的部分）

- **文本编码器**：T5Gemma2（26 层，隐藏维度 1152）+ 线性投影到主干空间。
- **主干**：Qwen3 风格解码器（28 层，隐藏维度 2048），每个 12.5 Hz 帧预测第一个 codebook
  token，音频 token 嵌入对 16 个 codebook 求和。
- **深度解码器**：12 层解码器，每帧生成 codebook 1–15，带独立 CFG。
- **音频分词器**：随附的 Qwen3-TTS 12 Hz 编解码器（Mimi 风格编码器 + 流式解码器），
  24 kHz，16 个 codebook。参考音频使用同一编解码器编码。
- 检查点内的 `codec_model`（仅供上游训练前向使用的 Mimi 副本）在推理时不构建。

eager 生成循环重新实现了官方 `FastBreezeStreamingRuntime`，并关闭所有快速
阶段（用 DynamicCache 替代 CUDA graph），以保持与 AIMDO 动态显存分页及 INT8 权重
转换的兼容性。使用 `decode_mode=cuda_graphs` 时，深度解码步骤会被捕获为
CUDA graph——捕获后的解码比走 AIMDO 分页的 eager 路径**快 7–9 倍**
（整段生成约快 3–4 倍）；该模式下模型权重常驻显存，不再由 AIMDO 分页。
官方 CUDA-graph 快速路径有意未移植。

## 许可证

节点包代码：Apache-2.0（见 `LICENSE`）。随附的模型代码保留其上游许可证
（breezeblue-ai/breeze-tts 与 Qwen/Qwen3-TTS 均为 Apache-2.0）。

**模型权重、衍生检查点与自托管输出受 BreezeBlue Research and Non-Commercial License
约束** — 见 [MODEL_LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)。
商业使用需获得 RESONIA, INC. 的书面授权。
