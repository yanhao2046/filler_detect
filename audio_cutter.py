#!/usr/bin/env python3
"""
Audio Cutter - Stage 3: Filler Removal via ffmpeg

Takes filler detection results + original audio, removes filler segments,
and outputs clean audio + summary report.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class KeepSegment:
    """A segment of audio to keep (between fillers)."""
    id: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def generate_keep_segments(
    fillers_json: Path,
    audio_duration: float,
    min_gap: float = 0.02
) -> List[KeepSegment]:
    """
    Generate keep-segment list from filler detection results.

    Args:
        fillers_json: Path to filler detection JSON (from detect_fillers)
        audio_duration: Total audio duration in seconds
        min_gap: Minimum segment duration to keep (seconds)

    Returns:
        List of KeepSegment objects
    """
    with open(fillers_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all delete-action marks, sorted by start time
    delete_marks = sorted(
        [m for m in data.get("marks", []) if m.get("suggested_action") == "delete"],
        key=lambda m: m["start"]
    )

    if not delete_marks:
        return [KeepSegment(id=0, start=0.0, end=audio_duration)]

    segments = []
    current_pos = 0.0

    for mark in delete_marks:
        filler_start = mark["start"]
        filler_end = mark["end"]

        # Keep the gap before this filler
        if filler_start - current_pos > min_gap:
            segments.append(KeepSegment(
                id=len(segments),
                start=current_pos,
                end=filler_start
            ))
        current_pos = filler_end

    # Keep the tail after last filler
    if audio_duration - current_pos > min_gap:
        segments.append(KeepSegment(
            id=len(segments),
            start=current_pos,
            end=audio_duration
        ))

    return segments


def cut_and_merge(
    audio_path: Path,
    keep_segments: List[KeepSegment],
    output_path: Path,
    codec: str = "libmp3lame",
    quality: str = "2"
) -> Path:
    """
    Cut audio into keep-segments and merge into clean output.

    Uses ffmpeg concat demuxer: cut each segment, then concatenate.

    Args:
        audio_path: Original audio file
        keep_segments: List of segments to keep
        output_path: Output clean audio path
        codec: Audio codec (libmp3lame for mp3, copy for fast but may glitch)
        quality: Audio quality (for libmp3lame: 0=best, 9=worst)

    Returns:
        Path to output audio file
    """
    if not keep_segments:
        raise ValueError("No keep segments to merge")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="filler_cut_") as tmpdir:
        tmpdir = Path(tmpdir)
        segment_files = []

        # Step 1: Cut each keep-segment
        for seg in keep_segments:
            seg_file = tmpdir / f"seg_{seg.id:04d}.mp3"
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(audio_path),
                '-ss', str(seg.start),
                '-t', str(seg.duration),
                '-acodec', codec,
                '-q:a', quality,
                str(seg_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg cut failed for segment {seg.id}: {result.stderr}"
                )
            segment_files.append(seg_file)

        # Step 2: Create concat list
        concat_list = tmpdir / "concat.txt"
        with open(concat_list, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        # Step 3: Merge
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg merge failed: {result.stderr}")

    return output_path


def generate_summary_report(
    audio_path: Path,
    output_audio: Path,
    fillers_json: Path,
    keep_segments: List[KeepSegment],
    output_path: Path
) -> Path:
    """
    Generate a markdown summary report.

    Args:
        audio_path: Original audio path
        output_audio: Clean audio path
        fillers_json: Filler detection JSON path
        keep_segments: Keep segments used for cutting
        output_path: Report output path (.md)

    Returns:
        Path to report file
    """
    # Load filler data
    with open(fillers_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    marks = data.get("marks", [])
    summary = data.get("summary", {})
    metadata = data.get("metadata", {})

    # Get durations
    original_duration = get_audio_duration(audio_path)
    clean_duration = get_audio_duration(output_audio)
    time_saved = original_duration - clean_duration

    # Count by type
    by_type = {}
    for mark in marks:
        text = mark.get("text", "?")
        by_type[text] = by_type.get(text, 0) + 1

    # Calculate delete rate
    total_fillers = len(marks)
    deleted = sum(1 for m in marks if m.get("suggested_action") == "delete")

    def fmt_time(seconds: float) -> str:
        """Format seconds to mm:ss or hh:mm:ss."""
        if seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}h {m}m {s}s"
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"

    # Build report
    lines = []
    lines.append(f"# FillerDetect Summary Report")
    lines.append(f"")
    lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Input**: `{audio_path.name}`")
    lines.append(f"- **Output**: `{output_audio.name}`")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Overall")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Original duration | {fmt_time(original_duration)} |")
    lines.append(f"| Clean duration | {fmt_time(clean_duration)} |")
    lines.append(f"| Time saved | **{time_saved:.1f}s** |")
    lines.append(f"| Total fillers | {total_fillers} |")
    lines.append(f"| Deleted | {deleted} |")
    lines.append(f"| Delete rate | **{deleted/total_fillers*100:.0f}%** |" if total_fillers > 0 else "| Delete rate | N/A |")
    lines.append(f"| Segments kept | {len(keep_segments)} |")
    lines.append(f"")
    lines.append(f"## Filler Types")
    lines.append(f"")
    lines.append(f"| Type | Count | % |")
    lines.append(f"|------|-------|---|")
    for text, count in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = count / total_fillers * 100 if total_fillers > 0 else 0
        lines.append(f"| {text} | {count} | {pct:.1f}% |")
    lines.append(f"")

    # Quality assessment
    if total_fillers > 0:
        rate = deleted / total_fillers * 100
        if rate >= 90:
            grade = "A (Excellent)"
        elif rate >= 80:
            grade = "B (Good)"
        elif rate >= 60:
            grade = "C (Acceptable)"
        else:
            grade = "D (Needs Review)"
    else:
        grade = "N/A (No fillers detected)"

    lines.append(f"## Assessment")
    lines.append(f"")
    lines.append(f"- **Grade**: {grade}")
    lines.append(f"- **Recommendation**: {'Ready to use' if deleted / max(total_fillers, 1) * 100 >= 90 else 'Manual review recommended'}")
    lines.append(f"")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path


def cut_fillers(
    audio_path: str,
    fillers_json: str,
    output_dir: str = "./output",
    output_filename: Optional[str] = None
) -> Dict:
    """
    Full Stage 3 pipeline: cut fillers from audio and generate summary report.

    This is the main entry point for Stage 3.

    Args:
        audio_path: Original audio file path
        fillers_json: Filler detection JSON path (from detect_fillers)
        output_dir: Output directory
        output_filename: Custom output filename (default: {stem}_clean.mp3)

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
    audio_path = Path(audio_path)
    fillers_json = Path(fillers_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not fillers_json.exists():
        raise FileNotFoundError(f"Fillers JSON not found: {fillers_json}")

    # Step 1: Get audio duration
    print(f"[Stage 3] Reading audio: {audio_path.name}")
    audio_duration = get_audio_duration(audio_path)

    # Step 2: Generate keep segments
    print(f"[Stage 3] Generating keep segments...")
    keep_segments = generate_keep_segments(fillers_json, audio_duration)
    print(f"  Keep segments: {len(keep_segments)}")

    # Step 3: Cut and merge
    if output_filename is None:
        output_filename = f"{audio_path.stem}_clean.mp3"
    output_audio = output_dir / output_filename

    print(f"[Stage 3] Cutting and merging ({len(keep_segments)} segments)...")
    cut_and_merge(audio_path, keep_segments, output_audio)

    # Step 4: Verify and get clean duration
    clean_duration = get_audio_duration(output_audio)
    time_saved = audio_duration - clean_duration

    # Step 5: Generate summary report
    report_path = output_dir / f"{audio_path.stem}_report.md"
    generate_summary_report(
        audio_path, output_audio, fillers_json,
        keep_segments, report_path
    )

    # Step 6: Print summary
    with open(fillers_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    fillers_removed = sum(
        1 for m in data.get("marks", [])
        if m.get("suggested_action") == "delete"
    )

    print(f"\n{'='*50}")
    print(f"Stage 3 Complete")
    print(f"{'='*50}")
    print(f"  Original:  {audio_duration:.1f}s ({audio_duration/60:.1f}min)")
    print(f"  Clean:     {clean_duration:.1f}s ({clean_duration/60:.1f}min)")
    print(f"  Saved:     {time_saved:.1f}s")
    print(f"  Fillers:   {fillers_removed} removed")
    print(f"  Output:    {output_audio}")
    print(f"  Report:    {report_path}")
    print(f"{'='*50}")

    return {
        "clean_audio": output_audio,
        "summary_report": report_path,
        "keep_segments": len(keep_segments),
        "fillers_removed": fillers_removed,
        "time_saved": round(time_saved, 1),
        "original_duration": round(audio_duration, 1),
        "clean_duration": round(clean_duration, 1)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python audio_cutter.py <audio.mp3> <fillers.json> [output_dir]")
        print("\nExample:")
        print("  python audio_cutter.py podcast.mp3 podcast_fillers.json ./output")
        sys.exit(1)

    audio = sys.argv[1]
    fillers = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "./output"

    result = cut_fillers(audio, fillers, out_dir)
