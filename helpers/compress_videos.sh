#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <input_dir> [output_dir]"
  echo "Example: $0 /path/to/videos /path/to/compressed_videos"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-${INPUT_DIR}/compressed}"

mkdir -p "$OUTPUT_DIR"

# Loop over mp4 files
shopt -s nullglob
for file in "$INPUT_DIR"/*.mp4; do
  filename=$(basename "$file")
  output_file="${OUTPUT_DIR}/${filename}"

  echo "Compressing $file -> $output_file"
  ffmpeg -i "$file" -vcodec libx264 -crf 28 -preset fast -acodec aac -b:a 128k "$output_file"
done

echo "All files compressed into: $OUTPUT_DIR"
