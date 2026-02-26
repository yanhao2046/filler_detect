<div align="center">

[English](README.md) | 中文

</div>

# FillerDetect

播客口癖识别与粗剪工具。自动检测中文播客中的填充词（呃、嗯、啊等），生成剪辑列表并输出干净音频。

## 功能

- VAD 智能分段（silero-vad）
- 高精度 ASR + 词级时间戳（Qwen3）
- 基于规则的口癖检测 + 置信度评分
- 可自定义口癖库
- 自动音频剪辑 + 总结报告
- 输出：JSON 标记、SRT 审核文件、Audacity 标签

## 性能

40 分钟播客音频测试结果：

| 指标 | 结果 |
|------|------|
| 检测口癖数 | 114 |
| 删除率 | **100%** |
| 节省时间 | 40.8 秒 |
| 处理耗时 | 约 40 秒 |

## 快速开始

```bash
pip install -r requirements.txt

# 完整流程：VAD分段 + Qwen3转录 + 口癖检测
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad
```

## 三阶段处理流程

### Stage 1：ASR 转录
VAD 将长音频切分为 <5 分钟的段，Qwen3-ASR 转录并生成词级时间戳，ForcedAligner 确保精确对齐。

```bash
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad
```

### Stage 2：口癖检测
模式匹配口癖库，计算置信度评分。

```python
from filler_detect import detect_fillers
result = detect_fillers('audio.json', './output', confidence_threshold=0.7)
```

### Stage 3：音频剪辑
自动移除口癖片段，合并输出干净音频。

```python
from filler_detect import cut_fillers
result = cut_fillers('podcast.mp3', 'podcast_fillers.json', './output')
# -> {"clean_audio": Path, "summary_report": Path, "time_saved": 40.8, ...}
```

```bash
# 命令行方式
python -m filler_detect.audio_cutter podcast.mp3 podcast_fillers.json ./output
```

## 支持的口癖类型

| 模式 | 类型 | 置信度 | 备注 |
|------|------|--------|------|
| 呃 | 单字填充 | 0.95 | 安全删除 |
| 嗯 | 单字填充 | 0.95 | 安全删除 |
| 啊 | 单字填充 | 0.85 | 句尾需保留 |
| 哦 | 单字填充 | 0.80 | 视上下文而定 |
| 然后 | 连接词 | 0.80 | 仅句首 |
| 就是 | 连接词 | 0.75 | 仅句首 |
| 那个 | 连接词 | 0.70 | 仅句首 |
| 所以 | 连接词 | 0.65 | 仅句首 |

可在 `filler_db.json` 中自定义口癖模式。

## ASR 选型历程

| 模型 | 时间戳精度 | 口癖检测 | 结论 |
|------|-----------|---------|------|
| FunASR | 漂移 ~5s/10min | 好 | 排除 |
| Whisper | 好 | 漏检口癖 | 排除 |
| SenseVoice | 无词级时间戳 | 好 | 排除 |
| **Qwen3** | **高精度** | **好** | **选用** |

关键发现：FunASR 的时间戳漂移导致精确剪辑不可能（删除率 0%）。Qwen3+ForcedAligner 通过高精度对齐解决了这个问题（删除率 100%）。

## 关联项目

**[PodTrans](https://github.com/yanhao2046/podtrans)** — 基于 FunASR 的播客转录工具。

PodTrans 和 FillerDetect 最初设计为两阶段管线。开发中 FunASR 时间戳漂移迫使 FillerDetect 自建了 Qwen3 ASR 方案。两个项目现在各自独立：

| 工具 | 适用场景 | ASR 引擎 |
|------|----------|----------|
| **PodTrans** | 转录、字幕、全文检索 | FunASR (paraformer-zh) |
| **FillerDetect** | 口癖剪辑、精确音频编辑 | Qwen3 + VAD |

两者的 JSON 输出格式兼容（相同的 segments + words 结构）。

## 项目结构

```
filler_detect/
├── filler_core.py      # 口癖检测核心（Stage 2）
├── audio_cutter.py     # 音频切割合并（Stage 3）
├── qwen3_pipeline.py   # Qwen3-ASR + ForcedAligner 流水线（Stage 1）
├── vad_segmenter.py    # VAD 智能分段
├── filler_db.json      # 口癖库配置
├── __init__.py         # 公开 API
├── requirements.txt    # Python 依赖
├── PRD.md              # 产品需求文档
├── WORKFLOW.md         # 详细工作流文档
└── CLAUDE.md           # 项目开发备注
```

## 环境要求

- Python 3.9+
- ffmpeg（系统依赖）
- 约 5GB 磁盘空间（模型）
- GPU 可选（支持 CUDA、Apple MPS、CPU 回退）
- transformers==4.57.6（qwen_asr 要求）

## 许可证

MIT
