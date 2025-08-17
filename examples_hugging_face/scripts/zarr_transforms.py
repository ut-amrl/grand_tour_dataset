from pathlib import Path
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
from matplotlib import cm
# This tutorial explains how to convert points and messages between coordinate frames using transformation matrices.

# Notation: We refer to a transform as having a parent frame and a child frame.
# The transform from parent to child is denoted as T_parent_to_child == T_target_to_source.
# This transform converts points from the source frame (parent) to the target frame (child).

# For example, suppose we have an odometry topic with parent "base" and child "odom".
# The transform T_base_to_odom allows us to convert a point given in the base frame (p_base) to the odom frame:

# ```p_odom = T_odom_to_base @ p_base```

# If you use right-side multiplication, the order of multiplication is reversed:
# ```p_odom = p_base @ np.linalg.inv(T_odom_to_base)```

# Using this notation, you can intuitively chain transforms:
# To convert from the camera frame to the odom frame:
# ```T_odom_to_base @ T_base_to_camera = T_odom_to_camera```


def inv(T_parent_to_child):
    """Invert a homogeneous transformation matrix."""
    rot = T_parent_to_child[:3, :3]
    t = T_parent_to_child[:3, 3]
    T_child_to_parent = np.eye(4)
    T_child_to_parent[:3, :3] = rot.T
    T_child_to_parent[:3, 3] = -rot.T @ t
    return T_child_to_parent


def pq_to_se3(p, q):
    """Convert position + quaternion into a 4x4 SE(3) matrix."""
    se3 = np.eye(4, dtype=np.float64)

    if isinstance(p, dict) and isinstance(q, dict):
        pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        quat = np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
    else:
        pos = np.asarray(p, dtype=np.float64)
        quat = np.asarray(q, dtype=np.float64)

    se3[:3, :3] = R.from_quat(quat).as_matrix()
    se3[:3, 3] = pos
    return se3


def attrs_to_se3(attrs):
    T_parent_to_child = pq_to_se3(attrs["transform"]["translation"], attrs["transform"]["rotation"])
    child = attrs["transform"]["base_frame_id"]
    parent = attrs["transform"]["child_frame_id"]
    return T_parent_to_child


def transform_points(points_parent, T_parent_to_child):
    """
    points: Nx3 array of points in the source frame
    transform_matrix: 4x4 transformation matrix from source to target frame
    """
    points_homo = np.hstack([points_parent, np.ones((points_parent.shape[0], 1))])
    points_child = (T_parent_to_child @ points_homo.T).T[:, :3]
    return points_child


def get_static_transform(mission_root, parent, child):
    # Works for all tf_statics

    t1 = mission_root["tf"].attrs["tf"]["box_base"]
    T_box_base_to_base = pq_to_se3(t1["translation"], t1["rotation"])  # this is verified

    if child != "base":
        t1 = mission_root["tf"].attrs["tf"][child]
        T_child_to_base = pq_to_se3(t1["translation"], t1["rotation"])

        # Be careful in current version some data is referenced with respect to box_base and some with respect to base!
        if t1["base_frame_id"] == "box_base":
            T_child_to_box_base = T_child_to_base
            T_child_to_base = T_child_to_box_base @ T_box_base_to_base
    else:
        T_child_to_base = np.eye(4)

    if parent != "base":
        t1 = mission_root["tf"].attrs["tf"][parent]
        T_parent_to_base = pq_to_se3(t1["translation"], t1["rotation"])

        # Be careful in current version some data is referenced with respect to box_base and some with respect to base!
        if t1["base_frame_id"] == "box_base":
            T_parent_to_box_base = T_parent_to_base
            T_parent_to_base = T_parent_to_box_base @ T_box_base_to_base

    else:
        T_parent_to_base = np.eye(4)

    T_parent_to_child = T_parent_to_base @ inv(T_child_to_base)
    return T_parent_to_child


def interpolate_pose_at_timestamp(timestamp, timestamps, pose_pos, pose_orien):
    idx = np.searchsorted(timestamps, timestamp)
    if idx == 0 or idx == len(timestamps):
        raise ValueError("Timestamp outside interpolation range.")
    if timestamps[idx] == timestamp:
        return pq_to_se3(pose_pos[idx], pose_orien[idx])

    idx1, idx2 = idx - 1, idx
    t1, t2 = timestamps[idx1], timestamps[idx2]

    # Position (linear)
    positions = np.vstack([pose_pos[idx1], pose_pos[idx2]])
    translation_interpolator = interp1d(
        [t1, t2], positions, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    interp_pos = translation_interpolator(timestamp)

    # Orientation (slerp)
    rotations = R.from_quat([pose_orien[idx1], pose_orien[idx2]])
    slerp = Slerp([t1, t2], rotations)
    interp_quat = slerp(timestamp).as_quat()

    return pq_to_se3(interp_pos, interp_quat)


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


class FastTfLookup:
    def __init__(self, odom_key, mission_root, parent, child):
        self.odom_key = odom_key
        self.mission_root = mission_root
        odom = mission_root[odom_key]
        self.odom = mission_root[odom_key]
        self.timestamps = odom["timestamp"][:]
        self.pose_pos = odom["pose_pos"][:]
        self.pose_orien = odom["pose_orien"][:]
        self.parent = parent
        self.child = child

    def __call__(self, timestamp: float, interpolate: bool = True, parent=None, child=None) -> np.ndarray:
        if self.timestamps[0] > timestamp or self.timestamps[-1] < timestamp:
            raise ValueError("Timestamp outside interpolation range.")

        idx = np.searchsorted(self.timestamps, timestamp)

        if not interpolate:
            T_parent_to_child = pq_to_se3(self.pose_pos[idx], self.pose_orien[idx])

        elif self.timestamps[idx] == timestamp:
            T_parent_to_child = pq_to_se3(self.pose_pos[idx], self.pose_orien[idx])
        else:
            idx1, idx2 = idx - 1, idx
            t1, t2 = self.timestamps[idx1], self.timestamps[idx2]

            # Position (linear)
            positions = np.vstack([pose_pos[idx1], pose_pos[idx2]])
            translation_interpolator = interp1d(
                [t1, t2], positions, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            interp_pos = translation_interpolator(timestamp)

            # Orientation (slerp)
            rotations = R.from_quat([self.pose_orien[idx1], self.pose_orien[idx2]])
            slerp = Slerp([t1, t2], rotations)
            interp_quat = slerp(timestamp).as_quat()

            T_parent_to_child = pq_to_se3(interp_pos, interp_quat)

        return self._transform_output_frames(T_parent_to_child, parent, child)

    def _transform_output_frames(self, T_parent_to_child, parent, child):
        if child is not None:
            T_child_to_child_new = get_static_transform(self.mission_root, self.child, child)
            T_parent_to_child_new = T_parent_to_child @ T_child_to_child_new
        else:
            T_parent_to_child_new = T_parent_to_child

        if parent is not None:
            T_parent_new_to_parent = get_static_transform(self.mission_root, parent, self.parent)
            T_parent_new_to_child_new = T_parent_new_to_parent @ T_parent_to_child_new
        else:
            T_parent_new_to_child_new = T_parent_to_child_new

        return T_parent_new_to_child_new


if __name__ == "__main__":
    MISSION_FOLDER = Path("~/grand_tour_dataset/2024-11-04-10-57-34").expanduser()
    mission_root = zarr.open_group(store=MISSION_FOLDER / "data", mode="r")

    # Example usage of FastTfLookup
    fast_tf_lookup = FastTfLookup("dlio_map_odometry", mission_root, parent="dlio_map", child="hesai_lidar")
    timestamp = 1730714272.07

    T_dlio_map_to_base = fast_tf_lookup(timestamp, interpolate=True, parent=None, child="base")
    T_dlio_map_to_hesai_lidar = fast_tf_lookup(timestamp, interpolate=True)

    T_hesai_lidar_to_base = get_static_transform(mission_root, "hesai_lidar", "base")
    T_base_to_hesai_lidar = get_static_transform(mission_root, "base", "hesai_lidar")
