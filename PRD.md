# FillerDetect - 口癖识别与粗剪工具 PRD

## 项目概述

### 目标
基于ASR转录结果，自动识别播客中的口癖（填充词、口头禅），标记删除建议，并输出带时间戳的干净文本，为后续音频剪辑提供数据支持。

### 核心要求
- 纯本地运行，无需联网
- 规则匹配为主，简单语义判断为辅
- 输出混合格式（原文+干净文本+标记）
- 支持一键批量删除建议
- 口癖库可自定义配置

---

## ASR选型全过程回顾

### 选型演进

| 阶段 | 方案 | 核心问题 | 删除率 | 结论 |
|------|------|---------|--------|------|
| **V1** | FunASR | 时间戳漂移~5s/10min | ~0% | 排除 |
| **V2** | FunASR闭环 | 漂移在闭环内抵消 | ~0%（实测失败） | 排除 |
| **V3** | Qwen3+VAD | 高精度时间戳+5分钟限制 | **100%** | **选用** |

### 各方案详细对比

| ASR | 时间戳精度 | 口癖识别 | 中文准确率 | 5分钟限制 | 结论 |
|-----|-----------|---------|-----------|----------|------|
| **FunASR** | ⚠️ 漂移 (~5s/10min) | ✅ 敏感 | ⭐⭐⭐⭐⭐ | 无 | 排除 |
| **Whisper** | ✅ 准确 | ❌ **漏检** | ⭐⭐⭐☆☆ | 无 | 排除 |
| **SenseVoice** | ❌ 无词级时间戳 | ✅ 敏感 | ⭐⭐⭐⭐☆ | 无 | 排除 |
| **WhisperX** | ✅ 强制对齐 | ❌ 漏检 | ⭐⭐⭐☆☆ | - | 排除 |
| **Qwen3** | ✅ **高精度** | ✅ **敏感** | ⭐⭐⭐⭐⭐ | **有** | **选用** |

### 关键问题分析

#### FunASR时间戳漂移问题

**现象**：
- FunASR 标记 "呃" 在 6:32
- 实际音频中 "呃" 在 6:00
- **漂移 32 秒**

**V2闭环方案的理论与现实的差距**：
```
理论：同一文件内漂移会抵消
真实音频：     [片段A][呃][片段B]
               0-100s 100-100.3s 100.3-600s

FunASR时间戳：  [片段A'][呃'][片段B']
               0-103s 103-103.3s 103.3-603s

问题：漂移不均匀！词级时间戳与实际音频位置错位
实际删除率：~0%
```

#### Qwen3方案的优势

1. **时间戳精度高**：ForcedAligner强制对齐，漂移极小
2. **对口癖敏感**：能识别"呃"、"嗯"、"啊"
3. **中文优化**：阿里Qwen3针对中文优化
4. **轻量级**：0.6B模型，本地可跑

**限制与解决**：
- **5分钟输入限制** → VAD智能分段解决

---

## V3技术方案：VAD + Qwen3

### 系统架构

```
长音频.mp3 (任意长度)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: VAD + Qwen3流水线                                   │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ VAD智能分段  │───>│ Qwen3-ASR    │───>│ ForcedAlign  │   │
│  │ silero-vad   │    │ 转录文本      │    │ 词级时间戳    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                                            │       │
│         │         每段独立处理，时间戳+offset校正       │       │
│         │                                            ▼       │
│         │                                   ┌──────────────┐  │
│         └──────────────────────────────────>│ 合并全局时间戳 │  │
│                                             └──────────────┘  │
│                                             audio.json        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Stage 2: 口癖检测 (filler_core.py)
    │
    ▼
Stage 3: 音频剪辑 (ffmpeg)
    │
    ▼
audio_clean.mp3
```

### VAD智能分段

#### 分段策略对比

| 方案 | 原理 | 优点 | 缺点 | 选型 |
|------|------|------|------|------|
| **VAD智能分段** | 检测语音停顿，在停顿处切分 | 语义完整，句首口癖不漏检 | 需VAD模型 | ✅ 选用 |
| **固定4:30+边界扩展** | 每段4分30秒，前后留余量 | 实现简单 | 可能切断句子 | ❌ 排除 |

#### VAD参数设计

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大段长 | 280秒 | 留20秒余量给Qwen3（5分钟=300秒） |
| 最小段长 | 10秒 | 避免过短片段 |
| 停顿阈值 | 0.3秒 | 视为句子边界 |
| 切分优先级 | 长停顿>句边界>强制切 | 保持语义完整 |

#### 时间戳对齐算法

```python
# VAD分段信息
segments = [
    {"id": 0, "start": 0.0, "end": 280.0, "offset": 0.0},
    {"id": 1, "start": 280.0, "end": 567.3, "offset": 280.0},
    ...
]

# 处理每段
for seg in segments:
    # 1. 切出音频段
    seg_audio = cut_audio(audio_path, seg.start, seg.end)

    # 2. Qwen3处理（段内时间戳，从0开始）
    words = qwen3_pipeline.process(seg_audio)

    # 3. 全局时间戳校正
    for w in words:
        global_words.append({
            "word": w.word,
            "start": w.start + seg.offset,  # 关键：加offset
            "end": w.end + seg.offset
        })
```

---

## 40分钟完整音频测试结果

### 测试配置
- **音频文件**：144-raw.mp3（39分47秒，2386.7秒）
- **模型**：Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B
- **VAD分段**：9段，平均265秒/段

### 测试结果

| 指标 | 数值 |
|------|------|
| VAD分段 | 9段 |
| Qwen3转录 | 完成（词级时间戳已修复） |
| 口癖检测 | 114个 |
| 删除率 | **100%** |
| 原始时长 | 2386.7秒（39.8分钟） |
| 剪辑后时长 | 2345.9秒（39.1分钟） |
| 节省时间 | **40.8秒** |

### 口癖类型分布

| 类型 | 数量 |
|------|------|
| 呃 (e_single) | 103个 |
| 嗯 (en_single) | 6个 |
| 啊 (a_single) | 5个 |

### 效果评估
- **评级**：A（效果优秀）
- **删除率**：100%（满足>90%目标）
- **音频质量**：剪辑后流畅，无明显卡顿

---

## 口癖库设计

```json
{
  "version": "1.0",
  "patterns": [
    {
      "id": "e_single",
      "pattern": "呃",
      "type": "filler_single",
      "match_mode": "exact",
      "position": "any",
      "base_confidence": 0.95,
      "description": "单字填充，安全删除"
    },
    {
      "id": "en_single",
      "pattern": "嗯",
      "type": "filler_single",
      "match_mode": "exact",
      "position": "any",
      "base_confidence": 0.95,
      "description": "单字填充"
    },
    {
      "id": "a_single",
      "pattern": "啊",
      "type": "filler_single",
      "match_mode": "exact",
      "position": "any",
      "base_confidence": 0.85,
      "description": "单字填充，句尾需保留"
    },
    {
      "id": "then_connector",
      "pattern": "然后",
      "type": "connector",
      "match_mode": "exact",
      "position": "sentence_start",
      "base_confidence": 0.80,
      "description": "句首连接词"
    },
    {
      "id": "just_connector",
      "pattern": "就是",
      "type": "connector",
      "match_mode": "exact",
      "position": "sentence_start",
      "base_confidence": 0.75,
      "description": "句首连接词"
    }
  ]
}
```

---

## 核心算法

### 口癖识别流程

```
输入: ASR转录JSON (含词级时间戳)
  ↓
1. 遍历每个segment的words
2. 对每个word匹配filler_db patterns
3. 检查位置约束（句首/句中/句尾）
4. 计算置信度 = base_confidence + position_bonus
5. 合并连续相同口癖（如"呃呃呃"）
6. 生成action标记
  ↓
输出: 混合格式JSON
```

### 置信度计算

```python
def calculate_confidence(word, pattern, context):
    score = pattern.base_confidence

    # 位置加分
    if pattern.position == "sentence_start" and is_sentence_start(word):
        score += scoring.position_bonus.sentence_start

    # 重复扣分（避免过度删除）
    if is_repeated_within_short_time(word, last_filler_time):
        score += scoring.repeat_penalty

    # 边界处理
    return min(1.0, max(0.0, score))
```

### 一键批量删除策略

```python
def suggest_bulk_delete(marks, min_confidence=0.8, min_count=3):
    """
    对相同pattern的口癖，如果：
    1. 出现次数 >= 3次
    2. 平均置信度 >= 0.8
    3. 标准差小（置信度稳定）

    则标记为 bulk_delete_eligible = true
    """
    pass
```

---

## 输出格式

### JSON 结构

```json
{
  "metadata": {
    "source": "144-raw.json",
    "duration": 2386.4,
    "processed_at": "2026-02-24T10:00:00",
    "model": "qwen3-asr+forced-aligner",
    "total_segments": 9
  },
  "summary": {
    "total_fillers": 114,
    "by_type": {
      "e_single": 103,
      "en_single": 6,
      "a_single": 5
    },
    "bulk_delete_candidates": 114,
    "estimated_time_saved": 40.8
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 280.0,
      "text": "...",
      "words": [
        {"word": "呃", "start": 17.04, "end": 17.20, "action": "delete"}
      ]
    }
  ],
  "marks": [
    {
      "id": "mark_001",
      "word": "呃",
      "start": 17.04,
      "end": 17.20,
      "action": "delete",
      "confidence": 0.95
    }
  ]
}
```

---

## 实现文件

| 文件 | 说明 |
|------|------|
| `vad_segmenter.py` | VAD智能分段模块 |
| `qwen3_pipeline.py` | Qwen3-ASR + ForcedAligner封装 |
| `filler_core.py` | 口癖检测核心 |
| `filler_db.json` | 口癖库配置 |
| `test-data/` | 测试数据索引 |

### 命令行接口

```bash
# VAD分段 + Qwen3处理（推荐用于长音频）
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad

# 仅VAD分段
python -m filler_detect.vad_segmenter audio.mp3 ./output --cut

# 检测口癖
python -c "from filler_detect import detect_fillers; detect_fillers('audio.json')"

# 完整流程（VAD + Qwen3 + 检测 + 剪辑）
filler-cut audio.mp3 --output clean.mp3 --asr qwen3 --vad
```

---

## 经验教训

### 关键发现

1. **时间戳精度是核心**：FunASR的漂移问题无法通过闭环完全解决，Qwen3的高精度对齐是关键
2. **VAD分段有效**：silero-vad检测句子边界准确，解决了Qwen3的5分钟限制
3. **offset校正简单可靠**：全局时间戳 = 段内时间戳 + offset，逻辑清晰
4. **API适配需谨慎**：qwen_asr返回的数据结构与transformers不同，需要实际检查字段名

### API集成踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| 词级时间戳为空 | ForcedAlignResult使用`items`而非`words`字段 | 修改字段名访问 |
| 时间戳全为0 | `start_time`/`end_time`已是秒，无需/1000 | 去掉单位转换 |
| ASR返回无text属性 | 返回的是ASRTranscription对象列表 | 取result[0].text |
| transformers版本冲突 | qwen_asr要求4.57.6 | 降级transformers |

### 各版本对比

| 维度 | V1/V2 (FunASR) | V3 (Qwen3+VAD) |
|------|---------------|----------------|
| 核心问题 | 时间戳漂移 | 5分钟限制 |
| 解决方案 | 闭环剪辑抵消 | VAD分段+offset校正 |
| 实际删除率 | ~0% | **100%** |
| 实现复杂度 | 简单 | 中等 |
| 适用场景 | 短音频快速处理 | 长音频精准剪辑 |

---

## 验收标准

### 功能验收

- [x] VAD智能分段实现（优先句子边界）
- [x] Qwen3-ASR转录（词级时间戳）
- [x] 口癖检测（114个口癖识别）
- [x] 音频剪辑（ffmpeg合并）
- [x] 40分钟音频验证（删除率100%）

### 质量验收

- [x] 口癖删除率 > 90%（实际100%）
- [x] 音频剪辑后流畅无卡顿
- [x] 时间戳精度满足剪辑需求

---

## 附录：接口定义

```python
def detect_fillers(
    input_json: Path,           # ASR转录结果
    output_dir: Path,           # 输出目录
    filler_db: Optional[Path] = None,  # 自定义口癖库
    confidence_threshold: float = 0.7,  # 置信度阈值
    enable_bulk_suggestion: bool = True  # 启用一键删除建议
) -> Dict:
    """
    口癖识别主函数

    Returns:
        {
            "output_json": Path,      # 主输出文件
            "summary_csv": Path,      # 统计摘要
            "total_fillers": int,
            "bulk_delete_candidates": int
        }
    """
    pass

def qwen3_transcribe(
    audio_path: Path,
    output_dir: Path,
    use_vad: bool = False
) -> Path:
    """
    Qwen3转录主函数

    Returns:
        Path to output JSON
    """
    pass
```

---

*最后更新：2026-02-25 - 40分钟完整音频测试完成，V3方案验证成功*
