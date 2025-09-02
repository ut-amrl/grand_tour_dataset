import zarr
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from grand_tour.zarr_transforms import FastTfLookup, inv
import plotly.io as pio

pio.renderers.default = "browser"

if __name__ == "__main__":
    mission = "2024-11-04-10-57-34"
    lidar_tags = [
        "velodyne_points_undistorted",
        "livox_points_undistorted",
        "hesai_points_undistorted",
    ]

    mission_folder = Path("~/grand_tour_dataset").expanduser() / mission
    mission_root = zarr.open_group(mission_folder / "data", mode="r")
    tf_lookup = FastTfLookup("anymal_state_odometry", mission_root, parent="base", child="odom")

    # pick a reference lidar for timestamps (e.g. velodyne)
    ref_tag = lidar_tags[0]
    ref_times = mission_root[ref_tag]["timestamp"][:]

    symbols = ["circle", "x", "diamond"]
    colors = ["red", "green", "blue"]

    for i in [100, 200, 300]:
        t_ref = ref_times[i]
        fig = go.Figure()
        all_points = []

        for j, lidar_tag in enumerate(lidar_tags):
            times = mission_root[lidar_tag]["timestamp"][:]
            idx = np.argmin(np.abs(times - t_ref))
            pts = mission_root[lidar_tag]["points"][idx, : mission_root[lidar_tag]["valid"][idx, 0]]

            # transform lidar points into odometry frame directly
            T_lidar_odom = inv(
                tf_lookup(times[idx], interpolate=True, parent=mission_root[lidar_tag].attrs["frame_id"])
            )
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
            pts_odom = (T_lidar_odom @ pts_h.T).T[:, :3]
            all_points.append(pts_odom)

            # choose different symbols for each lidar

            symbol = symbols[j % len(symbols)]
            color = colors[j % len(colors)]

            fig.add_trace(
                go.Scatter3d(
                    x=pts_odom[:, 0],
                    y=pts_odom[:, 1],
                    z=pts_odom[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        symbol=symbol,
                        color=color,
                        opacity=0.7,
                    ),
                    name=lidar_tag,
                )
            )

        # Ensure equal axis scaling
        all_points = np.vstack(all_points)
        x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
        y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
        z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
        max_range = (
            max(
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            )
            / 2.0
        )

        mid_x = np.mean(x_range)
        mid_y = np.mean(y_range)
        mid_z = np.mean(z_range)

        fig.update_layout(
            title=f"3D Lidar Fusion in Odom Frame @ frame {i}",
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Z [m]",
                xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
                yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
                zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
            ),
        )
        fig.show()
