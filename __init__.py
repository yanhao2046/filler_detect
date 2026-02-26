"""
FillerDetect - 口癖识别与粗剪 Skill

功能：
- 基于ASR转录结果识别口癖（填充词、口头禅）
- 标记删除建议，保留原文
- 输出带时间戳的干净文本
- 支持一键批量删除建议
- 自动剪辑音频，移除口癖片段
- 生成总结报告

用法：
    from filler_detect import detect_fillers, cut_fillers
    result = detect_fillers("transcript.json", output_dir="./output")
    cut_result = cut_fillers("podcast.mp3", result["output_json"], "./output")
"""

from pathlib import Path
from typing import Union, Dict, Optional, List

from .filler_core import (
    detect_fillers as _detect,
    parse_reviewed_srt as _parse_srt,
    generate_cut_list as _gen_cuts,
    generate_cut_statistics as _gen_stats,
    print_cut_statistics as _print_stats,
    FillerDetector
)
from .audio_cutter import (
    cut_fillers as _cut_fillers,
    generate_keep_segments as _gen_keep,
)


def detect_fillers(
    input_json: Union[str, Path],
    output_dir: Union[str, Path] = "./filler_output",
    db_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.7
) -> Dict:
    """
    检测音频转录中的口癖

    Args:
        input_json: ASR转录JSON路径（来自podtrans）
        output_dir: 输出目录
        db_path: 自定义口癖库JSON路径
        confidence_threshold: 置信度阈值（0-1），低于此值的标记为"review"

    Returns:
        {
            "output_json": Path,      # 检测结果文件路径
            "summary": {
                "total_fillers": int,
                "by_pattern": Dict,
                "bulk_delete_candidates": int,
                "estimated_time_saved": float
            }
        }
    """
    return _detect(input_json, output_dir, db_path, confidence_threshold)


def preview_fillers(
    input_json: Union[str, Path],
    max_preview: int = 10
) -> None:
    """
    预览口癖检测结果（不保存文件）

    Args:
        input_json: ASR转录JSON路径
        max_preview: 最多显示多少个口癖
    """
    detector = FillerDetector()
    result = detector.detect(input_json)

    print(f"\n=== 口癖预览（共{result.summary['total_fillers']}个）===")

    for i, mark in enumerate(result.marks[:max_preview]):
        print(f"{i+1}. [{mark.start:.2f}s-{mark.end:.2f}s] '{mark.text}' "
              f"({mark.type}, 置信度{mark.confidence:.2f})")

    if len(result.marks) > max_preview:
        print(f"... 还有 {len(result.marks) - max_preview} 个未显示")

    if result.bulk_delete_groups:
        print(f"\n可一键删除：")
        for group in result.bulk_delete_groups[:3]:
            print(f"  - '{group['text']}': {group['count']}次 "
                  f"(平均置信度{group['avg_confidence']})")


def parse_reviewed_srt(srt_path: Union[str, Path]) -> List[Dict]:
    """
    解析审核后的SRT文件，提取最终的删除决策

    Returns:
        [
            {
                "start": float,
                "end": float,
                "text": str,
                "action": "delete" | "keep",
                "original_tag": str
            }
        ]
    """
    return _parse_srt(Path(srt_path))


def generate_cut_list(reviewed_srt: Union[str, Path], output_path: Union[str, Path]) -> Path:
    """
    根据审核后的SRT生成剪辑指令

    Args:
        reviewed_srt: 审核后的SRT文件路径
        output_path: 输出剪辑列表路径
    """
    return _gen_cuts(Path(reviewed_srt), Path(output_path))


def generate_cut_statistics(
    original_audio: Union[str, Path],
    clean_audio: Union[str, Path],
    fillers_json: Union[str, Path],
    residual_fillers: Optional[List[Dict]] = None
) -> Dict:
    """
    生成粗剪效果统计报告

    Args:
        original_audio: 原始音频路径
        clean_audio: 剪辑后音频路径
        fillers_json: 口癖检测JSON路径
        residual_fillers: 残留口癖列表（可选）

    Returns:
        统计报告字典，包含:
        - overall: 总体统计（总口癖数、删除数、删除率、时长缩短等）
        - by_type: 口癖类型分布
        - residual_details: 残留口癖详情
        - assessment: 效果评估（质量评级、建议）
    """
    return _gen_stats(Path(original_audio), Path(clean_audio), Path(fillers_json), residual_fillers)


def print_cut_statistics(report: Dict) -> None:
    """
    打印粗剪效果统计报告

    Args:
        report: generate_cut_statistics 返回的报告字典
    """
    _print_stats(report)


def cut_fillers(
    audio_path: Union[str, Path],
    fillers_json: Union[str, Path],
    output_dir: Union[str, Path] = "./output",
    output_filename: Optional[str] = None
) -> Dict:
    """
    Stage 3: 从音频中移除口癖，生成干净音频 + 总结报告

    Args:
        audio_path: 原始音频文件路径
        fillers_json: 口癖检测JSON路径（detect_fillers 输出的 output_json）
        output_dir: 输出目录
        output_filename: 自定义输出文件名（默认: {stem}_clean.mp3）

    Returns:
        {
            "clean_audio": Path,
            "summary_report": Path,
            "keep_segments": int,
            "fillers_removed": int,
            "time_saved": float,
            "original_duration": float,
            "clean_duration": float
        }
    """
    return _cut_fillers(
        str(audio_path), str(fillers_json),
        str(output_dir), output_filename
    )


__all__ = [
    "detect_fillers", "preview_fillers", "parse_reviewed_srt",
    "generate_cut_list", "generate_cut_statistics", "print_cut_statistics",
    "cut_fillers", "FillerDetector"
]
