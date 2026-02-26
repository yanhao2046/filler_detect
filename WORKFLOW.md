# FillerDetect 工作流程文档

## 概述

本文档描述 filler_detect 完整的口癖粗剪工作流程，包括三个阶段：
1. **Stage 1**: ASR 转录（音频 → JSON）
2. **Stage 2**: 口癖检测（JSON → 标记）
3. **Stage 3**: 音频剪辑（标记 + 音频 → 干净音频）

## 架构设计

### 核心原则：闭环处理

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Stage 1    │     │  Stage 2    │     │  Stage 3    │
│  ASR转录    │ --> │  口癖检测   │ --> │  音频剪辑   │
│             │     │             │     │             │
│ audio.mp3   │     │ audio.json  │     │ keep.json   │
│     +       │     │     +       │     │     +       │
│ FunASR      │     │ filler_db   │     │ audio.mp3   │
│     |       │     │     |       │     │     |       │
│     v       │     │     v       │     │     v       │
│ audio.json  │     │ fillers.json│     │ audio_clean │
└─────────────┘     └─────────────┘     └─────────────┘
         ↑                                    |
         └──────── 同一音频文件闭环 ──────────┘
```

**关键**：所有时间戳操作都在**同一个音频文件**闭环内完成，时间戳漂移会自动抵消。

### ASR选型

| ASR | 时间戳精度 | 口癖识别 | 5分钟限制 | 推荐场景 |
|-----|-----------|---------|----------|---------|
| **FunASR** | 有漂移(~5s/10min) | 敏感 | 无 | 短音频、快速处理 |
| **Qwen3** | 高精度 | 敏感 | 有(5分钟) | 长音频、精准剪辑 |

**推荐**：音频>5分钟或需要精准剪辑时，使用 **Qwen3 + VAD分段**。

---

## Stage 1: ASR 转录

### 两种模式

#### 模式A: FunASR（默认）
- 适合音频 < 10分钟
- 速度快，无需分段
- 时间戳有漂移但闭环内不影响

#### 模式B: Qwen3 + VAD分段（推荐）
- 适合任意长度音频
- 时间戳精度高
- 需要VAD分段处理5分钟限制

### 功能
将音频文件转换为带词级时间戳的 JSON。

### 模式A: FunASR流程

#### 工具
- **podtrans skill** (FunASR)
- 模型: Paraformer-zh + SeACo + fa-zh (时间戳)

#### 输入输出
| 输入 | 输出 |
|------|------|
| `144-raw.mp3` | `144-raw.json` |

### 输出 JSON 格式
```json
{
  "metadata": {
    "duration": 2386.704,
    "language": "zh",
    "model": "paraformer-zh"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.63,
      "end": 2.83,
      "text": "你好，欢迎收听九五比特。",
      "words": [
        {"word": "你", "start": 0.63, "end": 0.79},
        {"word": "好", "start": 0.79, "end": 1.03},
        {"word": "，", "start": 1.03, "end": 1.15}
      ]
    }
  ]
}
```

### 命令行
```bash
python -m podtrans.cli 144-raw.mp3 --output ./output
```

### 模式B: Qwen3 + VAD分段流程（推荐）

适合长音频（>5分钟），时间戳精度高（94.9%删除率）。

#### 流程图
```
long_audio.mp3
    │
    ▼
┌─────────────────┐
│ VAD智能分段     │  silero-vad检测语音停顿
│ 最大280s/段     │  优先在句子边界切分
└─────────────────┘
    │
    ▼
segments: [{id:0, start:0, end:268}, ...]
    │
    ▼
┌────────────────────────────────────────┐
│ For each segment:                      │
│   1. Qwen3-ASR转录 → text              │
│   2. Qwen3-ForcedAligner对齐 → timestamps│
│   3. 时间戳 + offset → 全局时间戳        │
└────────────────────────────────────────┘
    │
    ▼
合并结果 ──> audio.json（与FunASR格式一致）
```

#### 工具
- **VAD**: silero-vad
- **ASR**: Qwen3-ASR-0.6B
- **对齐**: Qwen3-ForcedAligner-0.6B

#### 分段策略
| 参数 | 值 | 说明 |
|------|-----|------|
| 最大段长 | 280秒 | 留20秒余量给Qwen3 |
| 最小段长 | 10秒 | 避免过短片段 |
| 停顿阈值 | 0.3秒 | 视为句子边界 |
| 切分优先级 | 长停顿>句边界>强制切 | 保持语义完整 |

#### 命令行
```bash
# VAD分段 + Qwen3处理
python -m filler_detect.qwen3_pipeline 144-raw.mp3 ./output --vad

# 或仅VAD分段
python -m filler_detect.vad_segmenter 144-raw.mp3 ./output --cut
```

#### 时间戳对齐
```python
# 段内时间戳 + offset = 全局时间戳
segment_words = aligner.align(seg_audio, text)
global_words = [
    {
        "word": w.text,
        "start": w.start + segment.offset,  # 关键：加偏移
        "end": w.end + segment.offset
    }
    for w in segment_words
]
```

### 注意事项
- **时间戳漂移**: FunASR 存在 ~5秒/10分钟 的累积漂移
- **不影响结果**: 漂移是系统性的，闭环内自动抵消
- **Qwen3优势**: 时间戳精度高，94.9%删除率，适合精准剪辑

---

## Stage 2: 口癖检测

### 功能
从 ASR 结果中识别口癖词，生成保留段列表。

### 工具
- **filler_detect skill**
- 配置: `filler_db.json`

### 检测规则
| 口癖类型 | 示例 | 置信度 | 操作 |
|---------|------|--------|------|
| 单字填充 | 呃、嗯、啊 | 1.0 | 删除 |
| 句首连接 | 然后、就是、那个 | 0.6-0.8 | 暂不支持 |

### 中间数据

#### 1. 口癖标记 (`144-raw_fillers.json`)
```json
{
  "marks": [
    {
      "id": "mark_001",
      "text": "呃",
      "start": 18.37,
      "end": 18.61,
      "confidence": 1.0,
      "suggested_action": "delete"
    }
  ]
}
```

#### 2. 保留段列表 (`keep_segments.json`)
```json
[
  {"start": 0.0, "end": 18.37, "type": "keep"},
  {"start": 18.61, "end": 76.52, "type": "keep"},
  {"start": 76.68, "end": 110.04, "type": "keep"}
]
```

### 生成逻辑
```python
# 1. 收集所有口癖词时间戳
fillers = [word for word in words if word in filler_db]

# 2. 排序
fillers_sorted = sorted(fillers, key=lambda x: x['start'])

# 3. 生成保留段（口癖之间的片段）
keep_segments = []
current_pos = 0.0

for f in fillers_sorted:
    if f['start'] > current_pos:
        keep_segments.append({
            'start': current_pos,
            'end': f['start'],
            'type': 'keep'
        })
    current_pos = f['end']

# 4. 添加最后一段
if current_pos < audio_duration:
    keep_segments.append({
        'start': current_pos,
        'end': audio_duration,
        'type': 'keep'
    })
```

### 命令行
```bash
python -m filler_detect 144-raw.json --output ./output
```

---

## Stage 3: 音频剪辑

### 功能
根据保留段列表，切割并合并音频，删除口癖片段。

### 工具
- **ffmpeg**

### 处理流程
```
原始音频: [A][呃][B][嗯][C][啊][D]
          ↓
切割:     [A.mp3] [B.mp3] [C.mp3] [D.mp3]
          ↓
合并:     [A][B][C][D] = clean.mp3
```

### 实现细节

#### 1. 批量切割
```bash
# 为每个保留段切割音频
for seg in keep_segments:
    ffmpeg -i input.mp3 -ss seg.start -t (seg.end-seg.start) -c copy seg_N.mp3
```

#### 2. 生成 concat 列表
```
file 'seg_000.mp3'
file 'seg_001.mp3'
file 'seg_002.mp3'
...
```

#### 3. 无缝合并
```bash
ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp3
```

### 命令行
```bash
python -m filler_detect.cut 144-raw.mp3 --segments keep_segments.json --output clean.mp3
```

---

## 完整流程示例

### 场景：处理单集播客

```bash
# 1. 准备
mkdir -p ./output
INPUT="144-raw.mp3"

# 2. Stage 1: ASR 转录
python -m podtrans.cli $INPUT --output ./output
# 输出: ./output/144-raw.json

# 3. Stage 2: 口癖检测
python -m filler_detect ./output/144-raw.json --output ./output
# 输出: ./output/144-raw_fillers.json
#        ./output/keep_segments.json

# 4. Stage 3: 音频剪辑
python -m filler_detect.cut $INPUT \
    --segments ./output/keep_segments.json \
    --output ./output/144-clean.mp3

# 5. 验证
ls -lh ./output/144-clean.mp3
```

### 一键执行
```bash
filler-cut 144-raw.mp3 --output 144-clean.mp3
```

---

## 数据结构

### 数据流
```
audio.mp3
    ↓ (Stage 1: ASR)
audio.json ──> fillers.json
    ↓              ↓
    └──────┬──────┘
           ↓
     keep_segments.json
           ↓
    (Stage 3: Cut)
           ↓
    audio_clean.mp3
```

### 文件清单
| 文件 | 阶段 | 说明 |
|------|------|------|
| `audio.mp3` | Input | 原始音频 |
| `audio.json` | Stage 1 | ASR 转录结果 |
| `audio_fillers.json` | Stage 2 | 口癖检测详情 |
| `keep_segments.json` | Stage 2 | 保留段列表 |
| `audio_clean.mp3` | Stage 3 | 干净音频 |

---

## 性能指标

### 处理速度
| 阶段 | 40分钟音频 | 速度 |
|------|-----------|------|
| Stage 1 (ASR) | ~23秒 | ~100x 实时 |
| Stage 2 (检测) | ~1秒 | - |
| Stage 3 (剪辑) | ~15秒 | ~160x 实时 |
| **总计** | **~40秒** | **60x 实时** |

### 资源占用
| 资源 | 占用 |
|------|------|
| CPU | 高（ASR期间）|
| 内存 | ~5GB (FunASR) |
| 磁盘 | 原始音频 2x (临时片段) |

---

## 错误处理

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 口癖漏检 | ASR 未识别 | 接受少量漏检（粗剪）|
| 误删 | 非口癖被识别 | 提高置信度阈值 |
| 剪辑卡顿 | 片段边界问题 | 使用 ffmpeg `-c copy` |

### 回退方案
- 保留原始音频
- 保留 `keep_segments.json` 可重新剪辑
- 支持从任意阶段重新执行

---

## 后续扩展

### 可能的功能
1. **预览模式**: 播放口癖前后 3 秒音频
2. **白名单**: 标记不应删除的口癖
3. **批量处理**: 多集播客并行处理
4. **Web 界面**: 可视化口癖位置

### 与其他工具集成
- **Ferrite**: 导入剪辑列表精细调整
- **Descript**: 导出文本模式编辑
- **Reaper**: 生成 EDL 剪辑列表

---

## 附录

### 口癖数据库 (filler_db.json)
```json
{
  "patterns": [
    {
      "id": "e_single",
      "text": "呃",
      "type": "filler_single",
      "confidence": 1.0
    },
    {
      "id": "en_single",
      "text": "嗯",
      "type": "filler_single",
      "confidence": 1.0
    }
  ]
}
```

### 命令行完整参数
```bash
# Stage 1
podtrans audio.mp3 [--model paraformer-zh] [--device cpu/cuda]

# Stage 2
filler-detect audio.json [--threshold 0.7] [--db filler_db.json]

# Stage 3
filler-cut audio.mp3 --segments keep.json [--output clean.mp3]

# 一键
filler-cut audio.mp3 [--output clean.mp3] [--threshold 0.7]
```
