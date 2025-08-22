import zarr
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from grand_tour.zarr_transforms import inv, get_static_transform, FastTfLookup


def project_lidar_to_camera(points, K, T_lidar_to_cam, W, H, D=None, model="pinhole"):
    K = np.array(K, dtype=np.float32).reshape(3, 3)
    D = np.array(D, dtype=np.float32)
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    cam_pts = (T_lidar_to_cam @ pts_h.T).T[:, :3]

    if model == "pinhole":
        img_pts = (K @ cam_pts.T).T
        img_pts[:, :2] /= img_pts[:, 2:]
    else:
        obj = cam_pts.reshape(-1, 1, 3).astype(np.float32)
        rvec, tvec = np.zeros((3, 1)), np.zeros((3, 1))
        if model == "radtan":
            img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        elif model == "equidistant":
            img_pts, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        img_pts = img_pts.reshape(-1, 2)

    valid = (
        (cam_pts[:, 2] > 0) & (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
    )
    return img_pts[valid].astype(int), cam_pts[valid, 2]


def overlay_lidar_on_image(image, img_pts, depths, max_depth=30.0, cmap="inferno"):
    cmap = plt.get_cmap(cmap)
    depths = np.clip(depths, 0, max_depth)
    colors = (cmap(depths / max_depth)[:, :3] * 255).astype(np.uint8)
    overlay = image.copy()
    for (x, y), c in zip(img_pts, colors):
        cv2.circle(overlay, (x, y), 3, c.tolist(), -1)
    return overlay


if __name__ == "__main__":
    mission = "2024-11-04-10-57-34"
    image_tags = ["hdr_front", "hdr_left", "hdr_right"]
    lidar_tags = ["velodyne_points_undistorted"]  # "livox_points_undistorted", "hesai_points_undistorted",

    mission_folder = Path("~/grand_tour_dataset").expanduser() / mission
    mission_root = zarr.open_group(mission_folder / "data", mode="r")
    tf_lookup = FastTfLookup("anymal_state_odometry", mission_root, parent="base", child="odom")

    for lidar_tag in lidar_tags:
        for i in [100, 200, 300]:
            t_lidar = mission_root[lidar_tag]["timestamp"][i]
            pts = mission_root[lidar_tag]["points"][i, : mission_root[lidar_tag]["valid"][i, 0]]

            for img_tag in image_tags:
                # find closest image
                t_imgs = mission_root[img_tag]["timestamp"][:]
                idx = np.argmin(np.abs(t_imgs - t_lidar))
                img_path = mission_folder / "images" / img_tag / f"{idx:06d}.jpeg"
                image = np.array(Image.open(img_path))

                # transforms
                T_base_l = tf_lookup(t_lidar, interpolate=True, parent="base")
                T_base_c = tf_lookup(t_imgs[idx], interpolate=True, parent="base")
                T_l_c = (
                    T_base_c
                    @ inv(T_base_l)
                    @ get_static_transform(
                        mission_root, mission_root[lidar_tag].attrs["frame_id"], mission_root[img_tag].attrs["frame_id"]
                    )
                )

                # intrinsics
                cam_info = mission_root[img_tag].attrs["camera_info"]
                K, D, W, H, model = (
                    cam_info["K"],
                    cam_info["D"],
                    cam_info["width"],
                    cam_info["height"],
                    cam_info["distortion_model"],
                )

                img_pts, depths = project_lidar_to_camera(pts, K, inv(T_l_c), W, H, D, model)

                depths = np.log(depths)
                overlay = overlay_lidar_on_image(image, img_pts, depths, max_depth=np.log(15), cmap="turbo")

                dir_path = Path("debug") / "project_lidar_on_image" / mission
                dir_path.mkdir(parents=True, exist_ok=True)
                out_path = dir_path / f"overlay_{lidar_tag}_{img_tag}_{i:06d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
