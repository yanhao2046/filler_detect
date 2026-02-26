# 项目：FillerDetect - 中文播客口癖识别与粗剪

## 项目概述
基于 VAD + Qwen3-ASR + ForcedAligner 的口癖检测与音频粗剪工具。

---

## POC 验收标准

* **必须生成最终干净音频文件**，不能只停留在"检测完成"
* **验收指标必须是剪辑后音频的实际删除率**（不是检测到的口癖数量）
* **必须验证时间戳精度**：随机抽取 3-5 个口癖位置，人工确认剪辑准确性

---

## 测试数据规范

### 目录结构
```
/tmp/fillerdetect_[YYYYMMDD]/
├── input/          # 原始音频文件
│   └── 144-raw.mp3
└── output/         # 处理结果
    ├── 144-raw_qwen3.json
    ├── 144-raw_fillers.json
    ├── 144-raw_keep_segments.json
    └── 144-clean.mp3
```

### 规则
* 测试数据必须按日期分目录，禁止混用不同日期的中间产物
* 原始音频文件禁止修改，只读
* 每次完整测试必须清理 output 目录重新开始

---

## 第三方库使用规范

### qwen_asr 特别注意事项

* **ForcedAlignResult 字段是 `items` 不是 `words`**
* **时间戳单位是秒，无需除以 1000**
* **ASR 返回的是对象列表，取 `.text` 属性而非直接 `str()`**

**关键教训**：使用第三方库时，必须先用代码验证实际返回结构，禁止仅凭文档假设字段名/数据类型。

验证代码模板：
```python
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignResult, ForcedAlignItem
print(ForcedAlignResult.__dataclass_fields__)
print(ForcedAlignItem.__dataclass_fields__)

# 实际打印验证时间戳单位
result = aligner.align(audio, text, "Chinese")
print(result[0].items[0])  # 确认 start_time/end_time 的值范围
```

### transformers 版本
* 当前使用 4.57.6（qwen_asr 要求）
* 禁止升级到 5.x，会导致模型加载失败

---

## ASR 方案选型备忘

| 方案 | 时间戳精度 | 口癖识别 | 结论 |
|------|-----------|---------|------|
| FunASR | 漂移 ~5s/10min | 敏感 | ❌ 删除率 ~0% |
| Whisper | 准确 | **漏检** | ❌ 不适合 |
| Qwen3 + VAD | **高精度** | **敏感** | ✅ **选用** |

**关键教训**：时间戳漂移无法用"闭环抵消"解决，必须高精度对齐。

---

## VAD 分段参数

| 参数 | 值 | 说明 |
|------|-----|------|
| max_duration | 280s | Qwen3 限制 5min，留 20s 余量 |
| min_duration | 10s | 避免过短片段 |
| 停顿阈值 | 0.3s | 视为句子边界 |

---

## 口癖库配置

* 配置文件：`filler_db.json`
* 当前支持：`呃`、`嗯`、`啊`、`然后`、`就是`
* 置信度阈值：0.7

---

## POC 环境检查清单

运行完整测试前，必须确认：

```bash
# 1. 检查 Python 依赖
python3 -c "from qwen_asr import Qwen3ASRModel; print('✅ qwen_asr')"
python3 -c "import torch; print(f'✅ torch {torch.__version__}')"
python3 -c "import transformers; print(f'✅ transformers {transformers.__version__}')"

# 2. 检查 transformers 版本（必须为 4.57.6）
python3 -c "import transformers; assert transformers.__version__ == '4.57.6', '版本不匹配'"

# 3. 检查模型缓存（首次使用需下载约 1.2GB）
ls ~/.cache/modelscope/hub/Qwen/ 2>/dev/null || echo "⚠️ 首次运行将自动下载模型"

# 4. 检查 ffmpeg
ffmpeg -version | head -1
```

---

## 常见错误快速排查

| 错误 | 原因 | 解决 |
|------|------|------|
| 词级时间戳为空 | 字段名错误（items vs words） | 检查 qwen3_pipeline.py line 171 |
| 时间戳全为 0 | 单位错误（秒 vs 毫秒） | 去掉 /1000 转换 |
| ASR 返回无 text | 返回类型是对象 | 取 result[0].text |
| transformers 报错 | 版本冲突 | 降级到 4.57.6 |
| `qwen3_asr` 模型类型错误 | transformers 版本过高 | `pip install transformers==4.57.6` |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| vad_segmenter.py | VAD 智能分段 |
| qwen3_pipeline.py | Qwen3 ASR + ForcedAligner |
| filler_core.py | 口癖检测核心 |
| filler_db.json | 口癖库配置 |
| test-data/ | 测试数据索引 |
| PRD.md | 产品需求文档 |

---

*最后更新：2026-02-25 - 40分钟测试完成，删除率 100%*
