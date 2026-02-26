<div align="center">

English | [中文](README_zh.md)

</div>

# FillerDetect

Podcast filler word detection and rough-cut tool. Automatically identifies filler words (呃, 嗯, 啊, etc.) in Chinese podcasts and generates edit lists for removal.

## Features

- VAD-based intelligent audio segmentation (silero-vad)
- High-precision ASR with word-level timestamps (Qwen3)
- Rule-based filler word detection with confidence scoring
- Customizable filler word database
- Output: JSON markers, SRT review files, Audacity labels

## Performance

Tested on 40-minute podcast audio:

| Metric | Result |
|--------|--------|
| Filler words detected | 114 |
| Deletion rate | **100%** |
| Time saved | 40.8 seconds |
| Processing time | ~40 seconds |

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline: VAD segmentation + Qwen3 transcription + filler detection
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad
```

## Three-Stage Pipeline

### Stage 1: ASR Transcription
VAD splits long audio into segments (<5 min each), Qwen3-ASR transcribes with word-level timestamps, ForcedAligner ensures precise alignment.

```bash
python -m filler_detect.qwen3_pipeline audio.mp3 ./output --vad
```

### Stage 2: Filler Detection
Pattern matching against filler database with confidence scoring.

```python
from filler_detect import detect_fillers
result = detect_fillers('audio.json', './output', confidence_threshold=0.7)
```

### Stage 3: Audio Editing
Automatically remove filler segments and merge clean audio.

```python
from filler_detect import cut_fillers
result = cut_fillers('podcast.mp3', 'podcast_fillers.json', './output')
# -> {"clean_audio": Path, "summary_report": Path, "time_saved": 40.8, ...}
```

```bash
# Or via command line
python -m filler_detect.audio_cutter podcast.mp3 podcast_fillers.json ./output
```

## Supported Filler Words

| Pattern | Type | Confidence | Note |
|---------|------|------------|------|
| 呃 | Single filler | 0.95 | Safe to delete |
| 嗯 | Single filler | 0.95 | Safe to delete |
| 啊 | Single filler | 0.85 | Keep at sentence end |
| 哦 | Single filler | 0.80 | Context-dependent |
| 然后 | Connector | 0.80 | Sentence-start only |
| 就是 | Connector | 0.75 | Sentence-start only |
| 那个 | Connector | 0.70 | Sentence-start only |
| 所以 | Connector | 0.65 | Sentence-start only |

Custom patterns can be added to `filler_db.json`.

## ASR Model Selection History

| Model | Timestamp Accuracy | Filler Detection | Result |
|-------|-------------------|------------------|--------|
| FunASR | Drift ~5s/10min | Good | Excluded |
| Whisper | Good | Misses fillers | Excluded |
| SenseVoice | No word timestamps | Good | Excluded |
| **Qwen3** | **High precision** | **Good** | **Selected** |

Key finding: FunASR's timestamp drift (~5s per 10 min) made precise audio editing impossible (0% deletion rate). Qwen3+ForcedAligner solved this with high-precision alignment (100% deletion rate).

## Related Project

**[PodTrans](https://github.com/yanhao2046/podtrans)** — Podcast ASR transcription tool using FunASR.

PodTrans and FillerDetect were originally designed as a two-stage pipeline. During development, FunASR's timestamp drift required FillerDetect to build its own Qwen3-based ASR. Both projects now work independently:

| Tool | Best For | ASR Engine |
|------|----------|------------|
| **PodTrans** | Transcription, subtitles, full-text search | FunASR (paraformer-zh) |
| **FillerDetect** | Filler word removal, precise audio editing | Qwen3 + VAD |

Their JSON output formats are compatible (same segments + words structure).

## Project Structure

```
filler_detect/
├── filler_core.py      # Core detection engine (Stage 2)
├── audio_cutter.py     # Audio cutting and merging (Stage 3)
├── qwen3_pipeline.py   # Qwen3-ASR + ForcedAligner pipeline (Stage 1)
├── vad_segmenter.py    # VAD intelligent segmentation
├── filler_db.json      # Filler word database
├── __init__.py         # Public API
├── requirements.txt    # Python dependencies
├── PRD.md              # Product requirements document
├── WORKFLOW.md         # Detailed workflow documentation
└── CLAUDE.md           # Project-specific development notes
```

## Requirements

- Python 3.9+
- ffmpeg (system dependency)
- ~5GB disk space for models
- GPU optional (supports CUDA, Apple MPS, CPU fallback)
- transformers==4.57.6 (required by qwen_asr)

## License

MIT
