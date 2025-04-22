#!/usr/bin/env bash

folder="./"

# Function to display usage
# Function to display usage
usage() {
    echo "rosbag play warpper to easily select bags. Also sets use_sim_time to 1"
    echo "Usage: $0 [flags] [other rosbag pay args e.g. -r 0.5]"
    # echo "  --folder: Specify the folder path to search for bag files (default: current folder)"
    echo "  Flags:"
    echo "    --anymal          Add *_anymal_*.bag files"
    echo "    --gnss or --cpt7  Add cpt7_ie_rt.bag and cpt7_ie_tc.bag"
    echo "    --hdr             Add hdr_front.bag, hdr_left.bag, hdr_right.bag"
    echo "    --alphasense      Add alphasense.bag"
    echo "    --zed2i           Add zed2i_depth.bag, zed2i_images.bag, zed2i_vio.bag"
    echo "    --hesai           Add hesai.bag, hesai_packets.bag, hesai_undist.bag"
    echo "    --livox           Add livox.bag, livox_imu.bag, livox_undist.bag"
    echo "    --imus            Add all IMU-related bags"
    echo "    --dlio            Add dlio.bag"
    echo "    --depth_cameras   Add anymal_depth_cameras.bag"
    echo "    --ap20            Add ap20_imu.bag, ap20_prism_position.bag"
    echo "    --tf_minimal      Add tf_minimal.bag"
    echo "    --tf_model        Add tf_model.bag"
    exit 1
}

# Initialize variables
args=""
keys=()

# Define flag-to-key mappings
declare -A flag_to_keys=(
    [--anymal]="*anymal_command.bag *anymal_depth_cameras.bag *anymal_elevation.bag *anymal_imu.bag *anymal_locomotion.bag *anymal_state.bag *anymal_velodyne.bag *anymal_velodyne_undist.bag"
    [--gnss]="*cpt7_ie_rt.bag *cpt7_ie_tc.bag"
    [--cpt7]="*cpt7_ie_rt.bag *cpt7_ie_tc.bag"
    [--hdr]="*hdr_*.bag"
    [--cameras]="*hdr_*.bag *alphasense.bag"
    [--lidars]="*livox.bag *livox_undist.bag *hesai.bag *hesai_undist.bag"
    [--alphasense]="*alphasense.bag"
    [--zed2i]="*zed2i_depth.bag *zed2i_images.bag *zed2i_vio.bag"
    [--hesai]="*hesai.bag *hesai_undist.bag"
    [--livox]="*livox.bag *livox_imu.bag *livox_undist.bag"
    [--imus]="*adis.bag *stim320_imu.bag *alphasense_imu.bag *cpt7_imu.bag *ap20_imu.bag *anymal_imu.bag *livox_imu.bag"
    [--dlio]="*dlio.bag"
    [--depth_cameras]="*anymal_depth_cameras.bag"
    [--ap20]="*ap20_prism_position.bag *ap20_imu.bag"
    [--tf_minimal]="*tf_minimal.bag"
    [--tf_model]="*tf_model.bag"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        *)
            if [[ -n "${flag_to_keys[$1]}" ]]; then
                keys+=(${flag_to_keys[$1]})
            else
                args="$args $1"
            fi
            shift 1
            ;;
    esac
done


# Set ROS parameter for simulation time
rosparam set use_sim_time 1
# Play the filtered bag files
rosbag play --clock "${keys[@]}" $args