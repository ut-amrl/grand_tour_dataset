#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <subdir> <date> [rate]"
  echo "Example: $0 release_2024-11-11-12-07-40 2024-11-11-12-07-40 2.0"
  exit 1
fi

SUBDIR="$1"
DATE="$2"
RATE="${3:-5}"

# Hard-coded bag suffixes (everything after the date prefix)
BAGS=(
  "_alphasense.bag"
#  "_hdr_front.bag"
  "_anymal_state.bag"
  "_tf_minimal.bag"
  "_dlio.bag"
  "_tf_model.bag"
)

# Build full paths
BAG_PATHS=()
for suffix in "${BAGS[@]}"; do
  BAG_PATHS+=("${SUBDIR}/${DATE}${suffix}")
done

echo "Playing bags at rate ${RATE}x:"
printf '  %s\n' "${BAG_PATHS[@]}"

# Run rosbag play
rosbag play --rate "${RATE}" "${BAG_PATHS[@]}"

