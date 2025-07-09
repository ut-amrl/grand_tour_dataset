import kleinkram
from pathlib import Path
import rosbag
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

missions = kleinkram.list_missions(project_names=["GrandTourDataset"])

download_folder = Path("/tmp/grand_tour_data") # Change this to your desired download folder.

cfg = [
    # {
    #     "name": "hdr_front",
    #     "pattern": "*hdr_front.bag",
    #     "topic": "/boxi/hdr/front/image_raw/compressed",
    #     "distance": -1,
    #     "n_frames": 10
    # }, 
    {
        "name": "anymal_depth_rear_upper",
        "pattern": "*anymal_depth_cameras.bag",
        "topic": "/anymal/depth_camera/rear_upper/depth/image_rect_raw",
        "distance": -1,
        "n_frames": 10
    }, 
    {
        "name": "anymal_depth_front_upper",
        "pattern": "*anymal_depth_cameras.bag",
        "topic": "/anymal/depth_camera/front_upper/depth/image_rect_raw",
        "distance": -1,
        "n_frames": 10
    }, 
    {
        "name": "zed2i_depth",
        "pattern": "*zed2i_depth.bag",
        "topic": "/boxi/zed2i/depth/image_raw/compressed",
        "distance": -1,
        "n_frames": 10
    }, 
]

 # Ensure unique patterns
patterns = [c["pattern"] for c in cfg] + ["*_anymal_state.bag"]
patterns = list(set(patterns)) 

bridge = CvBridge()

invalid_mission_names = []

# Iterate over each mission in GrandTourDataset
for m in missions:
    current_folder = download_folder / m.name
    current_folder.mkdir(parents=True, exist_ok=True)

    # Attempt to download files matching the patterns
    files = kleinkram.list_files(mission_ids=[m.id], file_names=patterns)
    if len(files) != len(patterns):
        print(f"Mission {m.name} does not have a match for each pattern: {patterns}")
        invalid_mission_names.append(m.name)
        continue
    kleinkram.download(
        file_ids=[f.id for f in files],
        dest=current_folder,
        verbose=True,
        overwrite=True
    )

    # (optional) Loading state estimation bag for distance computation
    # state_bag_path = list(current_folder.glob(patterns[-1]))[0]
    # state_msgs = []
    # with rosbag.Bag(state_bag_path) as bag:
    #     for topic, msg, t in bag.read_messages(topics=["/anymal/state_estimator/pose"]):
    #         state_msgs.append(msg)


    # Iterate over all downloaded camera bags and extract images
    for c in cfg:
        camera_key = c["name"]
        camera_bag_path = list(current_folder.rglob(c["pattern"]))[0]
        (current_folder /  camera_key).mkdir(parents=True, exist_ok=True)

        with rosbag.Bag(camera_bag_path) as bag:
            n = 0
            for topic, img_msg, t in bag.read_messages(topics=[c["topic"]]):
                if n % c["n_frames"] != 0:
                    n += 1
                    continue

                if isinstance(img_msg, CompressedImage) or "compressed" in topic:
                    cv_image = bridge.compressed_imgmsg_to_cv2(img_msg)
                else:
                    cv_image = bridge.imgmsg_to_cv2(img_msg, img_msg.encoding)
                
                # Write image -> Currently uses header.seq for naming 
                image_filename = f"{camera_key}_{img_msg.header.seq:05d}.png"
                image_path = current_folder / camera_key / image_filename
                cv2.imwrite(str(image_path), cv_image)

                n += 1