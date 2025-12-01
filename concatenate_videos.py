"""
Script to concatenate multiple gameplay videos into a single video.
Uses ffmpeg to combine all episode videos in the results/breakout/videos/ directory.
"""

import subprocess
import os
from pathlib import Path


def concatenate_videos(
    video_dir="results/breakout/videos", output_name="breakout_full_gameplay.mp4"
):
    """
    Concatenate all gameplay videos in the specified directory.

    Args:
        video_dir: Directory containing video files
        output_name: Name of the output concatenated video
    """
    video_path = Path(video_dir)
    output_path = video_path / output_name

    # Find all video files and sort them
    video_files = sorted(video_path.glob("breakout_gameplay-episode-*.mp4"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} videos to concatenate:")
    for vf in video_files:
        print(f"  - {vf.name}")

    # Create a temporary file list for ffmpeg
    filelist_path = video_path / "filelist.txt"
    with open(filelist_path, "w") as f:
        for video_file in video_files:
            # Use absolute path and escape special characters
            f.write(f"file '{video_file.absolute()}'\n")

    # Use ffmpeg to concatenate videos
    print(f"\nConcatenating videos into: {output_path}")

    try:
        # Method 1: Concat demuxer (fast, no re-encoding)
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Successfully created: {output_path}")
        print(f"✓ File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

    except subprocess.CalledProcessError as e:
        print(f"Error with concat demuxer, trying re-encode method...")
        # Method 2: Re-encode (slower but more compatible)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(filelist_path),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "copy",
                    "-y",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Successfully created: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error concatenating videos: {e.stderr}")
            return

    finally:
        # Clean up temporary file
        if filelist_path.exists():
            filelist_path.unlink()

    print(f"\n✓ All done! Concatenated video saved to: {output_path}")


if __name__ == "__main__":
    concatenate_videos()
