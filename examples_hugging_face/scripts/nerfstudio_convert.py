import yaml
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import zarr
from tqdm import tqdm
from grand_tour.zarr_transforms import inv, get_static_transform, FastTfLookup, transform_points

ROS_CAMERA_TO_OPENCV_CAMERA = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)


def ros_to_gl_transform(transform_ros):
    transform_gl = transform_ros @ ROS_CAMERA_TO_OPENCV_CAMERA
    return transform_gl


def gl_to_ros_transform(transform_gl):
    transform_ros = transform_gl @ np.linalg.inv(ROS_CAMERA_TO_OPENCV_CAMERA)
    return transform_ros


class Masking:
    def __init__(self, nc):
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-coco-panoptic"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        (nc.output_folder / "mask").mkdir(parents=True, exist_ok=True)
        self.nc = nc

    def __call__(self, *args, **kwds):
        cv_image, image_path, invalid_mask, frame, timestamp, camera_tag = args
        mask_file_path = str(image_path).replace("rgb", "mask").replace(".jpeg", ".png")
        pil_image = Image.fromarray(cv_image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )
        segmentation_map = predicted_segmentation_maps[0]
        human_mask = (segmentation_map == 0).cpu().numpy()  # Person class is 0
        binary_mask = (~human_mask * 255).astype(np.uint8)

        # Apply the mask from the rectification
        binary_mask[invalid_mask] = 0

        # Save the binary mask as PNG
        Image.fromarray(binary_mask).convert("L").save(mask_file_path)

        # Logging percentage human pixels
        # human_pixel_count = np.sum(human_mask)
        # total_pixels = human_mask.size
        # coverage_percent = (human_pixel_count / total_pixels) * 100
        # print(f"Human coverage in frame: {coverage_percent:.2f}%")

        # Add metadata
        frame["mask_path"] = frame["file_path"].replace("rgb", "mask")
        return frame


class Depth:
    def __init__(self, nc):
        self.nc = nc
        (nc.output_folder / "depth").mkdir(parents=True, exist_ok=True)

        self.lidar_cfg = nc.config["lidars"]
        # preload lidar data
        for l_cfg in nc.config["lidars"]:
            l_cfg["points"] = self.nc.mission_root[l_cfg["tag"]]["points"][:]
            l_cfg["valid"] = self.nc.mission_root[l_cfg["tag"]]["valid"][:]
            l_cfg["timestamp"] = self.nc.mission_root[l_cfg["tag"]]["timestamp"][:]
            l_cfg["lidar_frame"] = self.nc.mission_root[l_cfg["tag"]].attrs["frame_id"]

    def __call__(self, *args, **kwds):
        cv_image, image_path, invalid_mask, frame, timestamp, camera_tag = args

        points_in_cam = self.get_n_lidar_points_in_camera_frame(timestamp, camera_tag)
        depth_image = self.project_lidar_to_camera(points_in_cam, timestamp, camera_tag)

        # Cleaning up
        invalid = depth_image == -1
        depth_image = depth_image.clip(0, 300)  # Clip to a reasonable max depth in meters
        depth_image = depth_image.astype(np.float32)
        depth_image[invalid] = 0.0

        # # Save depth image
        frame["depth_file_path"] = frame["file_path"].replace("rgb", "depth").replace(".jpg", ".png")
        cv2.imwrite(str(self.nc.output_folder / frame["depth_file_path"]), depth_image.astype(np.float32))

        self.overlay_depth_on_rgb(frame, "depth_overlay")
        return frame

    def project_lidar_to_camera(self, points_in_cam, timestamp, camera_tag):
        points_in_cam = np.concatenate(points_in_cam, axis=0)
        cam_info = self.nc.mission_root[camera_tag].attrs["camera_info"]
        H, W = cam_info["height"], cam_info["width"]
        K = self.nc.undist_helpers[camera_tag]["new_camera_matrix"]

        # Filter points behind the camera
        valid_points = points_in_cam[:, 2] > 0
        points_in_cam = points_in_cam[valid_points]

        if len(points_in_cam) == 0:
            return np.full((H, W), -1, dtype=np.float32)

        # Project to image plane
        image_points = (K @ points_in_cam.T).T
        image_points[:, 0] /= image_points[:, 2]
        image_points[:, 1] /= image_points[:, 2]

        # Filter points within image bounds
        valid_pixels = (
            (image_points[:, 0] >= 0) & (image_points[:, 0] < W) & (image_points[:, 1] >= 0) & (image_points[:, 1] < H)
        )

        valid_image_points = image_points[valid_pixels]
        valid_depths = points_in_cam[valid_pixels, 2]

        # Create depth image
        depth_image = np.full((H, W), -1, dtype=np.float32)

        if len(valid_image_points) > 0:
            pixel_coords = valid_image_points[:, :2].astype(int)
            for i, (w, h) in enumerate(pixel_coords):
                if 0 <= h < H and 0 <= w < W:
                    # Use closest depth if multiple points project to same pixel
                    if depth_image[h, w] == -1 or valid_depths[i] < depth_image[h, w]:
                        depth_image[h, w] = valid_depths[i]

        return depth_image

    def get_n_lidar_points_in_camera_frame(self, timestamp, camera_tag):
        T_base_to_odom__t_camera = self.nc.tf_lookup(timestamp, interpolate=True, parent="base")
        camera_frame = self.nc.mission_root[camera_tag].attrs["frame_id"]

        points = []
        for l_cfg in self.lidar_cfg:
            n = l_cfg["accumulate_scans"]
            indices = np.argpartition(np.abs(l_cfg["timestamp"] - timestamp), n)[:n]

            for i in indices:
                i = int(i)
                t_lidar = l_cfg["timestamp"][i]
                points_in_lidar = l_cfg["points"][i, : int(l_cfg["valid"][i, 0])]

                # Relative motion from different timestamps
                T_base_to_odom__t_lidar = self.nc.tf_lookup(t_lidar, interpolate=True, parent="base")
                T_t_lidar_to_t_cam = np.linalg.inv(T_base_to_odom__t_lidar) @ T_base_to_odom__t_camera

                T_lidar_to_cam = get_static_transform(self.nc.mission_root, l_cfg["lidar_frame"], camera_frame)
                T_lidar_to_cam = T_t_lidar_to_t_cam @ T_lidar_to_cam

                points_in_cam = transform_points(
                    points_in_lidar,
                    inv(T_lidar_to_cam),
                )
                points.append(points_in_cam)

        return points

    def overlay_depth_on_rgb(self, frame, debug_tag):
        # Required for visualization of depth
        from scipy.ndimage import grey_dilation
        import matplotlib.pyplot as plt

        rgb_image = np.array(Image.open(self.nc.output_folder / frame["file_path"]))
        depth_image = cv2.imread(str(self.nc.output_folder / frame["depth_file_path"]), cv2.IMREAD_UNCHANGED)
        # Normalize depth image for visualization - max range 10m
        depth_normalized = (depth_image.clip(0, 10.0) / 10.0 * 255).astype(np.uint8)

        # Dilate the depth image to increase pixel width to 3
        depth_normalized = grey_dilation(depth_normalized, size=(3, 3))

        # rgb_image H,W,3
        alpha = 0

        cmap = plt.get_cmap("turbo").reversed()
        color_depth = cmap(depth_normalized)  # H,W,4

        # Set alpha to 0 where depth is 0
        color_depth[..., 3] = np.where(depth_normalized == 0, 0, color_depth[..., 3])

        # Convert color_depth from float [0,1] to uint8 [0,255] and remove alpha channel
        color_depth_rgb = (color_depth[..., :3] * 255).astype(np.uint8)

        # Use alpha channel for blending: where alpha==0, keep rgb_image pixel
        alpha_mask = color_depth[..., 3][..., None]
        if len(rgb_image.shape) == 2:
            overlay = (alpha * rgb_image[:, :, None].repeat(3, axis=2) + (1 - alpha) * color_depth_rgb).astype(np.uint8)
            overlay = np.where(alpha_mask == 0, rgb_image[:, :, None].repeat(3, axis=2), overlay)
        else:
            overlay = (alpha * rgb_image + (1 - alpha) * color_depth_rgb).astype(np.uint8)
            overlay = np.where(alpha_mask == 0, rgb_image, overlay)

        # Convert overlay to BGR for cv2 if needed
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        tag = "Depth Overlay"
        text_size, _ = cv2.getTextSize(tag, font, font_scale, font_thickness)
        text_x = overlay_bgr.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(
            overlay_bgr,
            tag,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        # Save again with tag
        output_path = str(self.nc.output_folder / frame["depth_file_path"]).replace(".png", f"_overlay_{debug_tag}.png")
        cv2.imwrite(output_path, overlay_bgr)


class NerfstudioConverter:
    def __init__(self, config, mission_root: zarr.Group, mission_folder, output_folder, mission_name):
        self.config = config
        self.mission_root = mission_root
        self.mission_folder = mission_folder

        # anymal_state_odometry dlio_map_odometry
        self.tf_lookup = FastTfLookup("dlio_map_odometry", mission_root, parent="hesai_lidar", child="dlio_map")

        self.output_folder = output_folder / f"{mission_name}_nerfstudio"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.frames_json_file = self.output_folder / "transforms.json"
        self.images_folder = self.output_folder / "rgb"
        self.images_folder.mkdir(parents=True, exist_ok=True)

        self.image_counters = {key["tag"]: 0 for key in self.config["cameras"]}
        self.image_last_stored = {key["tag"]: 0 for key in self.config["cameras"]}

        self.plugins = []
        if self.config["create_mask_based_on_semantics"]:
            self.plugins.append(Masking(self))
        if self.config["create_depth_based_on_lidar"]:
            self.plugins.append(Depth(self))

        self.undist_helpers = {key["tag"]: {} for key in self.config["cameras"]}

    def undistort_image(self, image, config):
        K = np.array(self.mission_root[config["tag"]].attrs["camera_info"]["K"]).reshape((3, 3))
        D = np.array(self.mission_root[config["tag"]].attrs["camera_info"]["D"])
        h, w = image.shape[:2]

        helper = self.undist_helpers[config["tag"]]

        # Fill in auxiliary data for undistortion
        if not hasattr(helper, "new_camera_info"):
            if self.mission_root[config["tag"]].attrs["camera_info"]["distortion_model"] == "equidistant":
                helper["new_camera_matrix"] = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w, h), np.eye(3), balance=1.0, fov_scale=1.0
                )
                helper["D_new"] = [0, 0, 0, 0]
                helper["map1"], helper["map2"] = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), helper["new_camera_matrix"], (w, h), cv2.CV_16SC2
                )
            else:
                helper["new_camera_matrix"], _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                helper["D_new"] = [0, 0, 0, 0, 0]
                helper["map1"], helper["map2"] = cv2.initUndistortRectifyMap(
                    K, D, None, helper["new_camera_matrix"], (w, h), cv2.CV_16SC2
                )
            helper["invalid_mask"] = (
                cv2.remap(
                    np.ones(image.shape[:2], dtype=np.uint8),
                    helper["map1"],
                    helper["map2"],
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                == 0
            )

        undistorted_image = cv2.remap(
            image, helper["map1"], helper["map2"], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted_image

    def run(self):
        frames_data = {"camera_model": "OPENCV", "frames": []}

        for camera in self.config["cameras"]:
            camera_tag = camera["tag"]
            data = self.mission_root[camera_tag]
            timestamps = data["timestamp"][:]
            seqs = data["sequence_id"][:]
            last_t = None
            last_pos = None
            for i in tqdm(range(0, timestamps.shape[0]), desc=f"Processing {camera_tag}"):
                timestamp = timestamps[i]
                if last_t is not None and timestamp - last_t < 1 / camera["hz"] + 0.001:
                    continue
                last_t = timestamp

                try:
                    T_cam_to_odom__t_camera = self.tf_lookup(timestamp, interpolate=True, parent=camera_tag)
                except Exception as e:
                    continue

                if (
                    last_pos is not None
                    and np.linalg.norm(last_pos - T_cam_to_odom__t_camera[:2, 3]) < camera["distance_threshold"]
                ):
                    continue
                last_pos = T_cam_to_odom__t_camera[:2, 3]

                cv_image = cv2.imread(self.mission_folder / "images" / camera_tag / f"{i:06d}.jpeg")

                blur = cv2.Laplacian(cv_image, cv2.CV_64F).var()
                if blur < camera["blur_threshold"]:
                    print(f"Warning: Image too blurry (blur value: {blur}). Skipping.")
                    continue

                if self.image_counters[camera_tag] >= camera.get("max_images", float("inf")):
                    print(f"Skipping image {i} for camera {camera_tag} due to max limit.")
                    break

                self.image_counters[camera_tag] += 1
                image_filename = f"{camera_tag}_{seqs[i]:05d}.png"
                image_path = self.images_folder / image_filename

                cv_image = self.undistort_image(cv_image, camera)

                cv2.imwrite(str(image_path), cv_image)

                # Convert to OpenGL convention
                odom__cam__t_camera_gl = ros_to_gl_transform(T_cam_to_odom__t_camera)

                timestamp = timestamps[i]
                secs = int(timestamp)
                nsecs = int((timestamp - secs) * 1e9)

                K = self.undist_helpers[camera_tag]["new_camera_matrix"]
                D = self.undist_helpers[camera_tag]["D_new"]

                frame = {
                    "file_path": f"./rgb/{image_filename}",
                    "transform_matrix": odom__cam__t_camera_gl.tolist(),
                    "camera_frame_id": int(seqs[i]),
                    "fl_x": str(K[0, 0]),
                    "fl_y": str(K[1, 1]),
                    "cx": str(K[0, 2]),
                    "cy": str(K[1, 2]),
                    "w": str(data.attrs["camera_info"]["width"]),
                    "h": str(data.attrs["camera_info"]["height"]),
                    "k1": str(D[0]),
                    "k2": str(D[1]),
                    "p1": str(D[2]),
                    "p2": str(D[3]),
                    "timestamp": str(secs) + "_" + str(nsecs),
                }

                invalid_mask = self.undist_helpers[camera["tag"]]["invalid_mask"]
                for plugin in self.plugins:
                    frame = plugin(cv_image, image_path, invalid_mask, frame, timestamp, camera_tag)

                frames_data["frames"].append(frame)

        with open(self.frames_json_file, "w") as f:
            json.dump(frames_data, f, indent=2)


if __name__ == "__main__":
    CONFIG_FILE = Path("~/git/grand_tour_dataset/examples_hugging_face/grand_tour_release.yaml").expanduser()
    MISSION_FOLDER = Path("~/grand_tour_dataset/2024-11-04-10-57-34").expanduser()
    OUTPUT_FOLDER = Path("~/git/grand_tour_dataset/examples_hugging_face/data").expanduser()

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

    mission_root = zarr.open_group(store=MISSION_FOLDER / "data", mode="r")

    converter = NerfstudioConverter(
        config=config,
        mission_root=mission_root,
        mission_folder=MISSION_FOLDER,
        output_folder=OUTPUT_FOLDER,
        mission_name=MISSION_FOLDER.stem,
    )
    converter.run()
