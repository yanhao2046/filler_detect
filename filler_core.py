#!/usr/bin/env python3
"""
FillerDetect - 口癖识别核心模块

功能：
- 基于ASR转录结果识别口癖（填充词、口头禅）
- 标记删除建议，保留原文
- 输出带时间戳的干净文本
- 支持一键批量删除建议
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class WordMark:
    """词标记"""
    word: str
    start: float
    end: float
    action: str  # "keep" | "delete" | "review"
    confidence: float
    pattern_id: Optional[str] = None
    context: str = ""


@dataclass
class FillerMark:
    """口癖标记"""
    id: str
    segment_id: int
    word_index: int
    text: str
    start: float
    end: float
    type: str
    confidence: float
    suggested_action: str
    bulk_delete_eligible: bool = False


@dataclass
class Segment:
    """片段"""
    id: int
    start: float
    end: float
    original_text: str
    clean_text: str
    words: List[WordMark]
    fillers_in_segment: List[str]


@dataclass
class DetectionResult:
    """检测结果"""
    metadata: Dict
    summary: Dict
    segments: List[Segment]
    marks: List[FillerMark]
    bulk_delete_groups: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "original_text": s.original_text,
                    "clean_text": s.clean_text,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "action": w.action,
                            "confidence": w.confidence,
                            "pattern_id": w.pattern_id,
                            "context": w.context
                        }
                        for w in s.words
                    ],
                    "fillers_in_segment": s.fillers_in_segment
                }
                for s in self.segments
            ],
            "marks": [
                {
                    "id": m.id,
                    "segment_id": m.segment_id,
                    "word_index": m.word_index,
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                    "type": m.type,
                    "confidence": m.confidence,
                    "suggested_action": m.suggested_action,
                    "bulk_delete_eligible": m.bulk_delete_eligible
                }
                for m in self.marks
            ],
            "bulk_delete_groups": self.bulk_delete_groups
        }

    def save_json(self, output_path: Path) -> Path:
        """保存为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path

    def save_review_srt(self, output_path: Path) -> Path:
        """
        生成审核用SRT文件

        标记规则：
        - [DELETE] 高置信度，建议删除（红色/粗体）
        - [REVIEW] 低置信度，需人工确认（黄色）
        """
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        with open(output_path, "w", encoding="utf-8") as f:
            subtitle_idx = 1

            for seg in self.segments:
                # 收集这个片段中的所有口癖标记
                seg_marks = [m for m in self.marks if m.segment_id == seg.id]

                if not seg_marks:
                    # 无口癖的片段，正常输出
                    f.write(f"{subtitle_idx}\n")
                    f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
                    f.write(f"{seg.original_text}\n\n")
                    subtitle_idx += 1
                else:
                    # 有口癖，逐词输出带标记
                    current_text = ""
                    current_start = seg.start
                    last_end = seg.start

                    for word_idx, word in enumerate(seg.words):
                        # 检查这个词是否是口癖
                        mark = None
                        for m in seg_marks:
                            if m.word_index == word_idx:
                                mark = m
                                break

                        if mark:
                            # 先输出之前的普通文本
                            if current_text:
                                f.write(f"{subtitle_idx}\n")
                                f.write(f"{format_time(current_start)} --> {format_time(last_end)}\n")
                                f.write(f"{current_text}\n\n")
                                subtitle_idx += 1
                                current_text = ""

                            # 输出口癖标记（单独一行高亮）
                            tag = "[DELETE]" if mark.suggested_action == "delete" else "[REVIEW]"
                            f.write(f"{subtitle_idx}\n")
                            f.write(f"{format_time(word.start)} --> {format_time(word.end)}\n")
                            f.write(f"{tag} {word.word} (置信度:{mark.confidence:.2f})\n\n")
                            subtitle_idx += 1

                            current_start = word.end
                            last_end = word.end
                        else:
                            current_text += word.word
                            last_end = word.end

                    # 输出剩余文本
                    if current_text:
                        f.write(f"{subtitle_idx}\n")
                        f.write(f"{format_time(current_start)} --> {format_time(seg.end)}\n")
                        f.write(f"{current_text}\n\n")
                        subtitle_idx += 1

        return output_path

    def save_audacity_labels(self, output_path: Path) -> Path:
        """
        生成 Audacity Label Track 文件

        Audacity 标签格式:
            start_time\tend_time\tlabel_text

        使用方式:
            1. Audacity: File > Import > Labels...
            2. 选择生成的 .txt 文件
            3. 标签会显示在单独的 Label Track 上
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for mark in self.marks:
                label = f"[{mark.suggested_action.upper()}] {mark.text}"
                if mark.suggested_action == "delete":
                    label += " [DEL]"
                elif mark.suggested_action == "review":
                    label += " [REV]"
                f.write(f"{mark.start:.6f}\t{mark.end:.6f}\t{label}\n")

        return output_path


class FillerDetector:
    """口癖检测器"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化检测器

        Args:
            db_path: 口癖库JSON路径，默认使用内置配置
        """
        self.db_path = db_path or Path(__file__).parent / "filler_db.json"
        self.db = self._load_db()
        self.sentence_endings = {'。', '？', '！', '.', '?', '!', '…', '，', ','}

    def _load_db(self) -> Dict:
        """加载口癖库"""
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_sentence_start(self, words: List[Dict], word_index: int) -> bool:
        """判断是否为句首"""
        if word_index == 0:
            return True
        prev_word = words[word_index - 1].get("word", "")
        return prev_word in self.sentence_endings

    def _calculate_confidence(
        self,
        pattern: Dict,
        is_sentence_start: bool,
        word_duration: float
    ) -> float:
        """计算置信度"""
        score = pattern.get("base_confidence", 0.5)
        scoring = self.db.get("scoring", {})

        # 位置加分
        position = pattern.get("position", "any")
        if position == "sentence_start" and is_sentence_start:
            score += scoring.get("position_bonus", {}).get("sentence_start", 0.1)
        elif not is_sentence_start and position == "sentence_start":
            # 句首模式出现在句中，降低置信度
            score -= 0.2

        # 单字加分（更确定是填充词）
        if len(pattern.get("pattern", "")) == 1:
            score += scoring.get("short_word_bonus", 0.05)

        return min(1.0, max(0.0, score))

    def _match_pattern(self, word: str, pattern: Dict) -> bool:
        """匹配口癖模式"""
        pattern_text = pattern.get("pattern", "")
        match_mode = pattern.get("match_mode", "exact")

        if match_mode == "exact":
            return word == pattern_text
        elif match_mode == "contains":
            return pattern_text in word
        elif match_mode == "regex":
            return bool(re.match(pattern_text, word))
        return False

    def detect(
        self,
        input_json: Path,
        confidence_threshold: float = 0.7
    ) -> DetectionResult:
        """
        检测口癖

        Args:
            input_json: ASR转录JSON路径
            confidence_threshold: 置信度阈值

        Returns:
            DetectionResult 检测结果
        """
        # 加载ASR结果
        with open(input_json, "r", encoding="utf-8") as f:
            asr_data = json.load(f)

        metadata = {
            "source": str(input_json),
            "duration": asr_data.get("metadata", {}).get("duration", 0),
            "processed_at": datetime.now().isoformat(),
            "filler_db_version": self.db.get("version", "1.0"),
            "confidence_threshold": confidence_threshold
        }

        segments = []
        marks = []
        filler_counts = {}
        mark_counter = 0

        # 遍历每个片段
        for seg_idx, seg_data in enumerate(asr_data.get("segments", [])):
            words_data = seg_data.get("words", [])
            if not words_data:
                continue

            word_marks = []
            segment_fillers = []

            # 遍历每个字
            for word_idx, word_data in enumerate(words_data):
                word = word_data.get("word", "")
                start = word_data.get("start", 0)
                end = word_data.get("end", 0)

                # 尝试匹配口癖库
                matched_pattern = None
                is_sent_start = self._is_sentence_start(words_data, word_idx)

                for pattern in self.db.get("patterns", []):
                    if self._match_pattern(word, pattern):
                        position = pattern.get("position", "any")
                        if position == "sentence_start" and not is_sent_start:
                            continue
                        matched_pattern = pattern
                        break

                if matched_pattern:
                    confidence = self._calculate_confidence(
                        matched_pattern,
                        is_sent_start,
                        end - start
                    )

                    if confidence >= confidence_threshold:
                        action = "delete"
                    elif confidence >= confidence_threshold - 0.2:
                        action = "review"
                    else:
                        action = "keep"
                        matched_pattern = None

                    if matched_pattern and action in ("delete", "review"):
                        pattern_id = matched_pattern.get("id")
                        filler_type = matched_pattern.get("type")

                        word_mark = WordMark(
                            word=word,
                            start=start,
                            end=end,
                            action=action,
                            confidence=confidence,
                            pattern_id=pattern_id,
                            context="sentence_start" if is_sent_start else "sentence_middle"
                        )

                        mark_counter += 1
                        mark = FillerMark(
                            id=f"mark_{mark_counter:03d}",
                            segment_id=seg_idx,
                            word_index=word_idx,
                            text=word,
                            start=start,
                            end=end,
                            type=filler_type,
                            confidence=confidence,
                            suggested_action=action
                        )

                        word_marks.append(word_mark)
                        marks.append(mark)

                        if pattern_id:
                            filler_counts[pattern_id] = filler_counts.get(pattern_id, 0) + 1
                            segment_fillers.append(word)
                        continue

                # 非口癖词
                word_marks.append(WordMark(
                    word=word,
                    start=start,
                    end=end,
                    action="keep",
                    confidence=1.0
                ))

            # 构建片段文本
            original_text = "".join(w.word for w in word_marks)
            clean_text = "".join(w.word for w in word_marks if w.action != "delete")

            segment = Segment(
                id=seg_idx,
                start=seg_data.get("start", 0),
                end=seg_data.get("end", 0),
                original_text=original_text,
                clean_text=clean_text,
                words=word_marks,
                fillers_in_segment=segment_fillers
            )
            segments.append(segment)

        # 计算一键删除建议
        bulk_groups = self._calculate_bulk_delete(marks, filler_counts)

        # 更新marks的bulk_delete_eligible
        for group in bulk_groups:
            for mark_id in group.get("marks", []):
                for mark in marks:
                    if mark.id == mark_id:
                        mark.bulk_delete_eligible = True

        summary = {
            "total_fillers": len(marks),
            "by_pattern": filler_counts,
            "bulk_delete_candidates": sum(g.get("count", 0) for g in bulk_groups),
            "estimated_time_saved": round(sum(m.end - m.start for m in marks if m.suggested_action == "delete"), 1)
        }

        return DetectionResult(
            metadata=metadata,
            summary=summary,
            segments=segments,
            marks=marks,
            bulk_delete_groups=bulk_groups
        )

    def _calculate_bulk_delete(
        self,
        marks: List[FillerMark],
        filler_counts: Dict[str, int]
    ) -> List[Dict]:
        """计算一键删除建议"""
        groups = []
        context_rules = self.db.get("context_rules", {})
        min_count = context_rules.get("bulk_delete_min_count", 3)
        min_confidence = context_rules.get("bulk_delete_min_confidence", 0.8)

        # 按pattern分组统计
        pattern_stats = {}
        for mark in marks:
            pattern_id = mark.type
            if pattern_id not in pattern_stats:
                pattern_stats[pattern_id] = {
                    "marks": [],
                    "confidences": [],
                    "text": mark.text
                }
            pattern_stats[pattern_id]["marks"].append(mark.id)
            pattern_stats[pattern_id]["confidences"].append(mark.confidence)

        for pattern_id, stats in pattern_stats.items():
            count = len(stats["marks"])
            avg_confidence = sum(stats["confidences"]) / count if count > 0 else 0

            if count >= min_count and avg_confidence >= min_confidence:
                groups.append({
                    "pattern": pattern_id,
                    "text": stats["text"],
                    "count": count,
                    "avg_confidence": round(avg_confidence, 2),
                    "marks": stats["marks"],
                    "suggestion": "high_confidence_bulk_delete"
                })

        # 按count排序
        groups.sort(key=lambda x: -x["count"])
        return groups


def parse_reviewed_srt(srt_path: Path) -> List[Dict]:
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
    import re

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析SRT
    pattern = r'(\d+)\s+([\d:,]+)\s+-->\s+([\d:,]+)\s+(.+?)(?=\n\d+\s*\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    decisions = []
    for idx, start_str, end_str, text in matches:
        text = text.strip()

        # 检查是否有标记
        delete_match = re.match(r'\[DELETE\]\s*(.+)', text)
        review_match = re.match(r'\[REVIEW\]\s*(.+)', text)

        # 转换时间
        def parse_time(t):
            t = t.replace(',', '.')
            parts = t.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

        start = parse_time(start_str)
        end = parse_time(end_str)

        if delete_match:
            # 仍有DELETE标记 → 确认删除
            decisions.append({
                "start": start,
                "end": end,
                "text": delete_match.group(1).split('(')[0].strip(),
                "action": "delete",
                "original_tag": "DELETE"
            })
        elif review_match:
            # 仍有REVIEW标记 → 默认保留（用户没改）
            decisions.append({
                "start": start,
                "end": end,
                "text": review_match.group(1).split('(')[0].strip(),
                "action": "keep",
                "original_tag": "REVIEW"
            })
        else:
            # 普通字幕 → 保留
            decisions.append({
                "start": start,
                "end": end,
                "text": text,
                "action": "keep",
                "original_tag": None
            })

    return decisions


def generate_cut_list(reviewed_srt: Path, output_path: Path) -> Path:
    """
    根据审核后的SRT生成剪辑指令

    输出格式:
    ```
    # 需要删除的时间段
    0.63 - 0.85    # 呃
    2.89 - 3.10    # 嗯
    ...
    ```
    """
    decisions = parse_reviewed_srt(reviewed_srt)
    cuts = [d for d in decisions if d["action"] == "delete"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 口癖删除剪辑列表\n")
        f.write(f"# 生成时间: {datetime.now().isoformat()}\n\n")
        f.write(f"# 共 {len(cuts)} 处需要删除\n\n")

        for cut in cuts:
            f.write(f"{cut['start']:.2f} - {cut['end']:.2f}    # {cut['text']}\n")

    return output_path


def generate_cut_statistics(
    original_audio: Path,
    clean_audio: Path,
    fillers_json: Path,
    residual_fillers: Optional[List[Dict]] = None
) -> Dict:
    """
    生成粗剪效果统计报告

    Args:
        original_audio: 原始音频路径
        clean_audio: 剪辑后音频路径
        fillers_json: 口癖检测JSON路径
        residual_fillers: 残留口癖列表（人工标注或二次检测）

    Returns:
        统计报告字典
    """
    import subprocess

    # 获取音频时长
    def get_duration(audio_path: Path) -> float:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())

    # 加载口癖检测结果
    with open(fillers_json, 'r', encoding='utf-8') as f:
        fillers_data = json.load(f)

    marks = fillers_data.get('marks', [])
    total_fillers = len(marks)

    # 统计口癖类型
    by_type = {}
    for mark in marks:
        text = mark.get('text', '')
        by_type[text] = by_type.get(text, 0) + 1

    # 计算删除率
    deleted_count = total_fillers - len(residual_fillers) if residual_fillers else total_fillers
    delete_rate = (deleted_count / total_fillers * 100) if total_fillers > 0 else 0

    # 时长信息
    original_duration = get_duration(original_audio)
    clean_duration = get_duration(clean_audio)
    duration_reduction = original_duration - clean_duration

    # 残留口癖详情
    residual_details = []
    if residual_fillers:
        for r in residual_fillers:
            residual_details.append({
                'time': format_time(r['start']),
                'global_seconds': r['start'],
                'text': r['text'],
                'possible_reason': '靠近分段边界' if r.get('near_boundary') else '对齐误差'
            })

    report = {
        'overall': {
            'total_fillers': total_fillers,
            'deleted': deleted_count,
            'residual': len(residual_fillers) if residual_fillers else 0,
            'delete_rate': round(delete_rate, 1),
            'original_duration': round(original_duration, 1),
            'clean_duration': round(clean_duration, 1),
            'duration_reduction': round(duration_reduction, 1)
        },
        'by_type': by_type,
        'residual_details': residual_details,
        'assessment': {
            'quality': '优秀' if delete_rate >= 90 else '良好' if delete_rate >= 80 else '一般',
            'recommendation': '可直接使用' if delete_rate >= 90 else '建议人工复核残留'
        }
    }

    return report


def format_time(seconds: float) -> str:
    """格式化秒数为 mm:ss"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def print_cut_statistics(report: Dict):
    """打印粗剪效果统计"""
    ov = report['overall']

    print("\n" + "=" * 50)
    print("粗剪效果统计")
    print("=" * 50)

    print("\n【总体统计】")
    print(f"  总口癖数: {ov['total_fillers']}个")
    print(f"  成功删除: {ov['deleted']}个")
    print(f"  残留口癖: {ov['residual']}个")
    print(f"  删除率: {ov['delete_rate']}%")
    print(f"  原始时长: {format_time(ov['original_duration'])}")
    print(f"  剪辑后时长: {format_time(ov['clean_duration'])}")
    print(f"  时长缩短: {ov['duration_reduction']:.1f}秒")

    print("\n【口癖类型分布】")
    for text, count in sorted(report['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {text}: {count}个 ({count/ov['total_fillers']*100:.1f}%)")

    if report['residual_details']:
        print("\n【残留口癖详情】")
        for r in report['residual_details']:
            print(f"  {r['time']} ({r['global_seconds']:.1f}s) - {r['text']}")
            print(f"    可能原因: {r['possible_reason']}")

    print("\n【效果评估】")
    print(f"  质量评级: {report['assessment']['quality']}")
    print(f"  建议: {report['assessment']['recommendation']}")
    print("=" * 50)


def detect_fillers(
    input_json: Path,
    output_dir: Path,
    db_path: Optional[Path] = None,
    confidence_threshold: float = 0.7
) -> Dict:
    """
    口癖识别主函数

    Args:
        input_json: ASR转录JSON路径
        output_dir: 输出目录
        db_path: 自定义口癖库路径
        confidence_threshold: 置信度阈值

    Returns:
        {
            "output_json": Path,
            "summary": Dict
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FillerDetector(db_path)
    result = detector.detect(input_json, confidence_threshold)

    # 保存结果
    base_name = Path(input_json).stem
    output_path = output_dir / f"{base_name}_fillers.json"
    result.save_json(output_path)

    # 生成审核用SRT
    srt_path = output_dir / f"{base_name}_review.srt"
    result.save_review_srt(srt_path)

    # 生成剪辑列表（预览版）
    cut_path = output_dir / f"{base_name}_cuts.txt"
    generate_cut_list(srt_path, cut_path)

    # 生成 Audacity Label Track
    label_path = output_dir / f"{base_name}_audacity_labels.txt"
    result.save_audacity_labels(label_path)

    print(f"\n口癖检测完成：")
    print(f"  总口癖数：{result.summary['total_fillers']}")
    print(f"  按类型：{result.summary['by_pattern']}")
    print(f"  可一键删除：{result.summary['bulk_delete_candidates']}")
    print(f"  预计时长缩短：{result.summary['estimated_time_saved']:.1f}秒")
    print(f"\n输出文件：")
    print(f"  JSON: {output_path}")
    print(f"  审核SRT: {srt_path}")
    print(f"  剪辑列表: {cut_path}")
    print(f"  Audacity标签: {label_path}")
    print(f"\n审核流程（选一种）：")
    print(f"  方式1 - SRT字幕：用 Aegisub/VLC 打开 {srt_path.name}")
    print(f"  方式2 - Audacity：File > Import > Labels，选择 {label_path.name}")
    print(f"\n确认删除后运行: generate_cut_list('{srt_path.name}', 'final_cuts.txt')")

    return {
        "output_json": output_path,
        "review_srt": srt_path,
        "cut_list": cut_path,
        "audacity_labels": label_path,
        "summary": result.summary
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python filler_core.py <input_json> [output_dir]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else "./filler_output"

    detect_fillers(input_file, output_dir)
