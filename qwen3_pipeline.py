#!/usr/bin/env python3
"""
Qwen3-ASR + Qwen3-ForcedAligner 流水线

功能：
- Qwen3-ASR转录音频为文本
- Qwen3-ForcedAligner进行强制对齐，获取词级时间戳
- 支持VAD分段后的时间戳偏移校正
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# 导入VAD相关类
from .vad_segmenter import Segment


@dataclass
class Word:
    """词级对齐结果"""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class AlignmentResult:
    """对齐结果"""
    text: str
    words: List[Word]
    segment_id: int = 0
    offset: float = 0.0

    def with_offset(self, offset: float) -> "AlignmentResult":
        """返回应用偏移后的结果"""
        return AlignmentResult(
            text=self.text,
            words=[
                Word(
                    text=w.text,
                    start=w.start + offset,
                    end=w.end + offset,
                    confidence=w.confidence
                )
                for w in self.words
            ],
            segment_id=self.segment_id,
            offset=offset
        )


class Qwen3Pipeline:
    """Qwen3-ASR + ForcedAligner 流水线

    使用Qwen3模型进行ASR转录和强制对齐：
    1. Qwen3-ASR: 将音频转为文本
    2. Qwen3-ForcedAligner: 将文本与音频对齐，获取词级时间戳

    特点：
    - 时间戳精度高（漂移小）
    - 对口癖敏感（"呃"、"嗯"、"啊"能识别）
    - 5分钟输入限制（需配合VAD分段）
    """

    def __init__(
        self,
        asr_model_name: str = "Qwen/Qwen3-ASR-0.6B",
        aligner_model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        初始化流水线

        Args:
            asr_model_name: ASR模型名称
            aligner_model_name: 强制对齐模型名称
            device: 设备 ('cuda', 'mps', 'cpu')
            dtype: 数据类型
        """
        self.device = device or self._auto_device()
        self.dtype = dtype
        self.asr_model_name = asr_model_name
        self.aligner_model_name = aligner_model_name

        self.asr_model = None
        self.aligner = None

        self._load_models()

    def _auto_device(self) -> str:
        """自动选择设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_models(self):
        """加载模型"""
        try:
            # 加载Qwen3-ASR (使用qwen_asr包)
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

            print(f"Loading ASR model: {self.asr_model_name}")
            self.asr_model = Qwen3ASRModel.from_pretrained(
                self.asr_model_name,
                dtype=self.dtype,
                device_map=self.device
            )

            # 加载Qwen3-ForcedAligner
            print(f"Loading Aligner model: {self.aligner_model_name}")
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                self.aligner_model_name,
                dtype=self.dtype,
                device_map=self.device
            )

            print(f"Models loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Failed to load Qwen3 models: {e}\n"
                "Please install: pip install qwen-asr transformers torch"
            )

    def transcribe(self, audio_path: Path) -> str:
        """
        ASR转录

        Args:
            audio_path: 音频文件路径（建议<5分钟）

        Returns:
            转录文本
        """
        # 使用qwen_asr的API直接转录
        result = self.asr_model.transcribe(str(audio_path))
        # qwen_asr返回的是ASRTranscription列表
        if isinstance(result, list) and len(result) > 0:
            return result[0].text.strip()
        return str(result).strip()

    def align(self, audio_path: Path, text: str) -> AlignmentResult:
        """
        强制对齐

        Args:
            audio_path: 音频文件路径
            text: 转录文本

        Returns:
            AlignmentResult 包含词级时间戳
        """
        # 使用qwen_asr的API直接对齐
        alignment_results = self.aligner.align(str(audio_path), text, "Chinese")

        # 转换为Word对象列表
        words = []
        # qwen_asr返回的是ForcedAlignResult列表
        if alignment_results and len(alignment_results) > 0:
            align_result = alignment_results[0]
            # ForcedAlignResult的字段是'items'，不是'words'
            if hasattr(align_result, 'items') and align_result.items:
                for w in align_result.items:
                    # ForcedAlignItem字段: text, start_time, end_time
                    words.append(Word(
                        text=getattr(w, 'text', str(w)),
                        start=float(getattr(w, 'start_time', 0)),  # 已经是秒
                        end=float(getattr(w, 'end_time', 0)),
                        confidence=1.0  # ForcedAlignItem没有confidence字段
                    ))

        return AlignmentResult(
            text=text,
            words=words
        )

    def _parse_alignment_outputs(
        self,
        outputs: torch.Tensor,
        original_text: str
    ) -> AlignmentResult:
        """
        解析对齐输出

        Qwen3-ForcedAligner输出格式:
        - 词之间有时间戳标记如 <time_123> (表示1.23秒)
        """
        # 解码token
        decoded = self.aligner_processor.batch_decode(
            outputs,
            skip_special_tokens=False
        )[0]

        # 解析词和时间戳
        words = []
        current_time = 0.0

        import re

        # 提取所有token
        tokens = decoded.split()

        for token in tokens:
            # 检查是否是时间戳
            time_match = re.match(r'<time_(\d+)>', token)
            if time_match:
                # 时间戳: time_123 -> 1.23秒
                current_time = int(time_match.group(1)) / 100.0
            else:
                # 普通词
                token = token.strip()
                if token and token not in ['<pad>', '</s>', '<s>']:
                    # 估算词时长（粗略估计每个字0.2-0.4秒）
                    word_duration = len(token) * 0.25
                    word = Word(
                        text=token,
                        start=current_time,
                        end=current_time + word_duration,
                        confidence=1.0
                    )
                    words.append(word)
                    current_time += word_duration

        return AlignmentResult(
            text=original_text,
            words=words
        )

    def process_segment(
        self,
        audio_path: Path,
        segment_id: int = 0,
        offset: float = 0.0
    ) -> AlignmentResult:
        """
        处理单个音频段（完整流水线）

        Args:
            audio_path: 段音频路径
            segment_id: 段ID
            offset: 全局时间偏移（用于校正时间戳）

        Returns:
            带全局时间戳的对齐结果
        """
        # 1. ASR转录
        text = self.transcribe(audio_path)

        if not text:
            return AlignmentResult(text="", words=[], segment_id=segment_id, offset=offset)

        # 2. 强制对齐
        result = self.align(audio_path, text)
        result.segment_id = segment_id

        # 3. 应用偏移校正
        if offset > 0:
            result = result.with_offset(offset)

        return result

    def process_long_audio(
        self,
        audio_path: Path,
        segments: List[dict],
        output_dir: Path
    ) -> Path:
        """
        处理长音频（配合VAD分段）

        Args:
            audio_path: 原始音频路径
            segments: VAD分段信息 [{id, start, end, offset}, ...]
            output_dir: 输出目录

        Returns:
            合并后的JSON路径
        """
        from .vad_segmenter import VADSegmenter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_words = []
        all_segments = []

        segmenter = VADSegmenter()

        for seg_info in segments:
            seg_id = seg_info['id']
            offset = seg_info['offset']

            print(f"\n处理段 {seg_id}: {seg_info['start']:.1f}s - {seg_info['end']:.1f}s")

            # 切出音频段
            seg_audio = output_dir / f"temp_seg_{seg_id:03d}.wav"
            segmenter.cut_audio_segment(
                audio_path,
                Segment(**seg_info),
                seg_audio
            )

            # 处理
            result = self.process_segment(seg_audio, seg_id, offset)

            # 收集结果
            seg_words = [
                {
                    'word': w.text,
                    'start': w.start,
                    'end': w.end,
                    'confidence': w.confidence,
                    'segment_id': seg_id
                }
                for w in result.words
            ]
            all_words.extend(seg_words)

            all_segments.append({
                'id': seg_id,
                'start': seg_info['start'],
                'end': seg_info['end'],
                'text': result.text,
                'words': seg_words
            })

            # 清理临时文件
            seg_audio.unlink(missing_ok=True)

        # 生成统一格式JSON（与FunASR格式一致）
        output = {
            'metadata': {
                'source': str(audio_path),
                'total_segments': len(segments),
                'model': 'qwen3-asr+forced-aligner'
            },
            'segments': all_segments,
            'words': all_words
        }

        output_path = output_dir / f'{audio_path.stem}_qwen3.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n处理完成: {output_path}")
        print(f"总词数: {len(all_words)}")

        return output_path


def qwen3_transcribe(
    audio_path: Path,
    output_dir: Path,
    use_vad: bool = False
) -> Path:
    """
    Qwen3转录主函数

    Args:
        audio_path: 输入音频
        output_dir: 输出目录
        use_vad: 是否使用VAD分段（音频>5分钟时推荐）

    Returns:
        JSON结果路径
    """
    from .vad_segmenter import segment_audio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Qwen3Pipeline()

    if use_vad:
        # VAD分段
        vad_result = segment_audio(audio_path, output_dir, cut_audio=False)
        segments = [
            {
                'id': seg.id,
                'start': seg.start,
                'end': seg.end,
                'offset': seg.offset
            }
            for seg in vad_result['segments']
        ]

        # 分段处理
        return pipeline.process_long_audio(audio_path, segments, output_dir)
    else:
        # 直接处理
        result = pipeline.process_segment(audio_path)

        output = {
            'metadata': {
                'source': str(audio_path),
                'model': 'qwen3-asr+forced-aligner'
            },
            'segments': [{
                'id': 0,
                'start': 0,
                'end': max(w.end for w in result.words) if result.words else 0,
                'text': result.text,
                'words': [
                    {'word': w.text, 'start': w.start, 'end': w.end}
                    for w in result.words
                ]
            }]
        }

        output_path = output_dir / f'{audio_path.stem}_qwen3.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return output_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python qwen3_pipeline.py <audio.mp3> [output_dir] [--vad]")
        print("  --vad: 启用VAD分段（处理长音频）")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else './qwen3_output'
    use_vad = '--vad' in sys.argv

    result_path = qwen3_transcribe(audio_file, out_dir, use_vad=use_vad)
    print(f"\n结果保存至: {result_path}")
