---
name: filler_detect
description: 播客口癖识别与粗剪工具。自动检测"呃"、"嗯"、"啊"等填充词，标记删除建议，输出剪辑列表。当用户说"检测口癖"、"删除口头禅"、"filler detection"、"清理播客"时触发。
argument-hint: "[audio_file_or_json] [output_dir]"
---

# FillerDetect - 口癖识别与粗剪

自动检测播客中的口癖（呃、嗯、啊等），标记删除建议，生成剪辑列表。

## 使用方式

```bash
# 完整流程：VAD分段 + Qwen3转录 + 口癖检测
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad

# 仅口癖检测（已有ASR结果）
python -c "from filler_detect import detect_fillers; detect_fillers('audio.json', './output')"
```

## 三阶段处理流程

1. **Stage 1: ASR转录** — Qwen3-ASR + VAD智能分段，生成词级时间戳
2. **Stage 2: 口癖检测** — 规则匹配 + 置信度计算，标记删除/保留
3. **Stage 3: 音频剪辑** — ffmpeg 切割合并，输出干净音频

## 核心指标

- 口癖删除率：100%（40分钟测试音频）
- 支持口癖：呃、嗯、啊、哦、然后、就是、那个、所以
- 处理速度：40分钟音频约40秒完成

## 已知限制

- Qwen3 单次输入限制 5 分钟，通过 VAD 分段解决
- 连接词（然后、就是等）检测需要句首约束，置信度较低
- 系统依赖 ffmpeg（音频剪辑阶段需要）
