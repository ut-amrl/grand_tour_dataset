from pathlib import Path
import numpy as np
import zarr
from grand_tour.zarr_transforms import (
    get_static_transform,
    pq_to_se3,
    attrs_to_se3,
    FastTfLookup,
    interpolate_pose_at_timestamp,
)

# This tutorial explains how to convert points and messages between coordinate frames using transformation matrices.

# Notation: We refer to a transform as having a parent frame and a child frame.
# The transform from parent to child is denoted as T_parent_to_child.

# For example, suppose we have an odometry topic with parent "odom" and child "base".
# The transform T_odom_to_base is a 4x4 SE3 transformation and allows us to convert a point in homogenous coordinates given in the base frame (p_base) to the odom frame:

# ```p_odom = T_odom_to_base @ p_base```

# Using this notation, you can intuitively chain transforms:
# To convert from the camera frame to the odom frame:
# ```T_odom_to_base @ T_base_to_camera = T_odom_to_camera```


if __name__ == "__main__":
    MISSION_FOLDER = Path("~/grand_tour_dataset/2024-11-04-10-57-34").expanduser()
    mission_root = zarr.open_group(store=MISSION_FOLDER / "data", mode="r")

    T_box_base_to_hdr_front = attrs_to_se3(mission_root["hdr_front"].attrs)

    # Example 1 usage:
    # This method aligns with <rosrun tf tf_echo parent child>!
    tf = get_static_transform(mission_root, "box_base", "base")
    # <rosrun tf tf_echo box_base base>

    # Example 2 usage:
    tf = get_static_transform(mission_root, "base", "hdr_right")
    # <rosrun tf tf_echo base hdr_right>

    tf = get_static_transform(mission_root, "hdr_left", "hdr_right")
    # <rosrun tf tf_echo hdr_left hdr_right>
    print(tf)

    # Each topic maps between two frames, e.g., "odom" to "base".
    # The topic arrive at different rates.
    odom_to_base = mission_root["anymal_state_odometry"]
    dlio_world_to_hesai_lidar = mission_root["dlio_map_odometry"]

    # Preloading all data speeds up tf_lookups.
    timestamps = dlio_world_to_hesai_lidar["timestamp"][:]
    pose_pos = dlio_world_to_hesai_lidar["pose_pos"][:]
    pose_orien = dlio_world_to_hesai_lidar["pose_orien"][:]

    # Example get closest timestamp
    desired_timestamp = 1730714272.07
    idx = np.argmin(np.abs(timestamps - desired_timestamp))
    T_dlio_world_to_hesai_lidar = pq_to_se3(pose_pos[idx], pose_orien[idx])

    # Can be easily combined with the static transforms:
    T_hesai_lidar_to_hdr_front = get_static_transform(mission_root, "hesai_lidar", "hdr_front")
    T_dlio_world_to_hdr_front = T_dlio_world_to_hesai_lidar @ T_hesai_lidar_to_hdr_front

    T_dlio_world_to_hesai_lidar_inter = interpolate_pose_at_timestamp(
        desired_timestamp, timestamps, pose_pos, pose_orien
    )

    # Example usage of FastTfLookup
    fast_tf_lookup = FastTfLookup("dlio_map_odometry", mission_root, parent="dlio_map", child="hesai_lidar")
    timestamp = 1730714272.07

    T_dlio_map_to_base = fast_tf_lookup(timestamp, interpolate=True, parent=None, child="base")
    T_dlio_map_to_hesai_lidar = fast_tf_lookup(timestamp, interpolate=True)

    T_hesai_lidar_to_base = get_static_transform(mission_root, "hesai_lidar", "base")
    T_base_to_hesai_lidar = get_static_transform(mission_root, "base", "hesai_lidar")
