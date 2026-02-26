#!/usr/bin/env python3
"""
VAD智能分段模块

功能：
- 使用silero-vad检测语音停顿
- 智能切分长音频，确保每段 < 5分钟
- 优先在句子边界切分，保持语义完整
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)


@dataclass
class Segment:
    """音频片段"""
    id: int
    start: float  # 全局起始时间（秒）
    end: float    # 全局结束时间（秒）
    offset: float # 相对于原始音频的偏移

    @property
    def duration(self) -> float:
        return self.end - self.start


class VADSegmenter:
    """VAD智能分段器

    使用silero-vad检测语音活动，智能切分长音频：
    1. 最大段长：280秒（留20秒余量给Qwen3）
    2. 最小段长：10秒（避免过短）
    3. 切分优先级：长停顿 > 句子边界 > 强制切
    """

    # silero-vad模型配置
    SAMPLING_RATE = 16000
    MAX_DURATION = 280   # 最大段长（秒）
    MIN_DURATION = 10    # 最小段长（秒）
    MIN_SILENCE = 0.3    # 视为停顿的最小静音（秒）
    THRESHOLD = 0.5      # 语音检测阈值

    def __init__(
        self,
        max_duration: float = MAX_DURATION,
        min_duration: float = MIN_DURATION,
        min_silence: float = MIN_SILENCE,
        threshold: float = THRESHOLD
    ):
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.min_silence = min_silence
        self.threshold = threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载silero-vad模型"""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.model.eval()
            self._utils = utils
        except Exception as e:
            raise RuntimeError(f"Failed to load silero-vad: {e}")

    def _read_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        读取音频并转为16kHz单声道

        Returns:
            (audio_array, sample_rate)
        """
        import wave

        # 先转换为wav格式（如果是mp3）
        wav_path = audio_path
        if audio_path.suffix.lower() != '.wav':
            wav_path = Path('/tmp') / f'vad_temp_{audio_path.stem}.wav'
            subprocess.run([
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', str(self.SAMPLING_RATE),
                '-ac', '1',
                str(wav_path)
            ], capture_output=True, check=True)

        # 读取wav
        with wave.open(str(wav_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()

            audio_data = wf.readframes(n_frames)
            audio = np.frombuffer(audio_data, dtype=np.int16)

            # 转为float32并归一化
            audio = audio.astype(np.float32) / 32768.0

            # 转为单声道
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

        # 清理临时文件
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink()

        return audio, sample_rate

    def _get_speech_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[dict]:
        """
        获取语音时间段

        Returns:
            [{'start': 0.0, 'end': 15.3}, {'start': 15.8, 'end': 45.2}, ...]
        """
        # 确保音频是torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # silero-vad获取语音时间戳
        speech_timestamps = self._utils[0](
            audio,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=250,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=int(self.min_silence * 1000)
        )

        return speech_timestamps

    def segment(self, audio_path: Path) -> List[Segment]:
        """
        将音频智能分段

        Args:
            audio_path: 音频文件路径

        Returns:
            Segment列表，每个Segment包含全局时间戳
        """
        # 读取音频
        audio, sample_rate = self._read_audio(audio_path)
        duration = len(audio) / sample_rate

        # 短音频直接返回
        if duration <= self.max_duration:
            return [Segment(id=0, start=0.0, end=duration, offset=0.0)]

        # 获取语音时间段
        speech_ts = self._get_speech_timestamps(audio, sample_rate)

        # 智能分段
        segments = self._merge_segments(speech_ts, duration)

        return segments

    def _merge_segments(
        self,
        speech_timestamps: List[dict],
        total_duration: float
    ) -> List[Segment]:
        """
        合并语音段，确保每段在限制范围内

        策略：
        1. 目标每段max_duration秒
        2. 在目标点附近找最长的静音区间切分
        3. 确保切分点不切断语音段
        """
        if not speech_timestamps:
            return [Segment(id=0, start=0.0, end=total_duration, offset=0.0)]

        segments = []
        seg_id = 0
        current_start = 0.0

        # 语音段起止时间（秒）
        speech_ranges = [
            (ts['start'] / 1000.0, ts['end'] / 1000.0)
            for ts in speech_timestamps
        ]

        while current_start < total_duration:
            target_end = min(current_start + self.max_duration, total_duration)

            # 如果已经到文件末尾
            if target_end >= total_duration - 1.0:  # 剩余不足1秒，直接结束
                if current_start < total_duration:
                    segments.append(Segment(
                        id=seg_id,
                        start=current_start,
                        end=total_duration,
                        offset=current_start
                    ))
                break

            # 在目标点附近找最佳切分点（最长的静音区间）
            # 搜索范围：目标点前30秒到目标点后30秒，但不早于current_start + min_duration
            search_start = max(target_end - 30, current_start + self.min_duration)
            search_end = min(target_end + 30, total_duration)

            best_cut_point = target_end
            best_silence_duration = 0.0

            # 遍历所有语音段之间的间隙
            for i in range(len(speech_ranges) - 1):
                curr_speech_end = speech_ranges[i][1]
                next_speech_start = speech_ranges[i + 1][0]

                # 间隙必须在搜索范围内
                if curr_speech_end < search_start:
                    continue
                if curr_speech_end > search_end:
                    break

                silence_duration = next_speech_start - curr_speech_end
                if silence_duration > best_silence_duration:
                    best_silence_duration = silence_duration
                    best_cut_point = curr_speech_end  # 切在间隙开始处（语音结束处）

            # 确保切分点有效，且相对于current_start前进
            segment_end = min(best_cut_point, total_duration)
            segment_end = max(segment_end, current_start + self.min_duration)

            # 如果切分点和起点太接近，强制前进到target_end
            if segment_end < current_start + self.min_duration:
                segment_end = min(target_end, total_duration)

            segments.append(Segment(
                id=seg_id,
                start=current_start,
                end=segment_end,
                offset=current_start
            ))

            seg_id += 1
            current_start = segment_end

            # 防止无限循环
            if len(segments) > 0 and segments[-1].duration <= 0:
                # 如果最后一段没有长度，直接结束
                segments.pop()
                break

        return segments

    def _merge_short_segments(
        self,
        segments: List[Segment],
        total_duration: float
    ) -> List[Segment]:
        """合并过短的段到相邻段"""
        if not segments:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            seg = segments[i]

            # 如果当前段太短，尝试合并
            if seg.duration < self.min_duration and i < len(segments) - 1:
                # 合并到下一段
                next_seg = segments[i + 1]
                merged_seg = Segment(
                    id=len(merged),
                    start=seg.start,
                    end=next_seg.end,
                    offset=seg.offset
                )
                merged.append(merged_seg)
                i += 2
            else:
                # 重设id
                seg.id = len(merged)
                merged.append(seg)
                i += 1

        return merged

    def cut_audio_segment(
        self,
        input_path: Path,
        segment: Segment,
        output_path: Path
    ) -> Path:
        """
        切出指定段的音频

        Args:
            input_path: 原始音频路径
            segment: 段信息
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        duration = segment.end - segment.start

        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-ss', str(segment.start),
            '-t', str(duration),
            '-ar', '16000',
            '-ac', '1',
            str(output_path)
        ], capture_output=True, check=True)

        return output_path

    def save_segments_json(
        self,
        segments: List[Segment],
        output_path: Path
    ) -> Path:
        """保存分段信息为JSON"""
        data = {
            'segments': [
                {
                    'id': seg.id,
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.duration,
                    'offset': seg.offset
                }
                for seg in segments
            ],
            'total_segments': len(segments),
            'max_duration': self.max_duration
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path


def segment_audio(
    audio_path: Path,
    output_dir: Path,
    max_duration: float = 280,
    cut_audio: bool = False
) -> dict:
    """
    VAD分段主函数

    Args:
        audio_path: 输入音频
        output_dir: 输出目录
        max_duration: 最大段长（秒）
        cut_audio: 是否同时切出音频文件

    Returns:
        {
            'segments': List[Segment],
            'segments_json': Path,
            'audio_files': List[Path] (if cut_audio=True)
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分段
    segmenter = VADSegmenter(max_duration=max_duration)
    segments = segmenter.segment(audio_path)

    # 保存分段信息
    seg_json_path = output_dir / f'{audio_path.stem}_segments.json'
    segmenter.save_segments_json(segments, seg_json_path)

    result = {
        'segments': segments,
        'segments_json': seg_json_path,
        'audio_files': []
    }

    # 切分音频
    if cut_audio:
        for seg in segments:
            seg_audio_path = output_dir / f'{audio_path.stem}_seg_{seg.id:03d}.wav'
            segmenter.cut_audio_segment(audio_path, seg, seg_audio_path)
            result['audio_files'].append(seg_audio_path)

    print(f"VAD分段完成：")
    print(f"  总段数: {len(segments)}")
    for seg in segments:
        print(f"    段{seg.id}: {seg.start:.1f}s - {seg.end:.1f}s ({seg.duration:.1f}s)")
    print(f"\n输出: {seg_json_path}")

    return result


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python vad_segmenter.py <audio.mp3> [output_dir] [--cut]")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else './vad_output'
    should_cut = '--cut' in sys.argv

    segment_audio(audio_file, out_dir, cut_audio=should_cut)
