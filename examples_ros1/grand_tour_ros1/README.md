# GrandTour ROS1

These examples demonstrate how to work with the GrandTour dataset in ROS1, including downloading rosbags, launching sensors, and replaying data.

## Python Examples

### 1-) Visualizing and Extracting Extrinsic Transforms

The GrandTour dataset contains calibration information for all sensors in the form of transforms. These examples show how to visualize and extract this information.

#### Visualizing Transforms

To visualize the transform tree between all sensors in the dataset:

```bash
rosrun rqt_tf_tree rqt_tf_tree
```

#### Extracting Transforms

You can extract transforms between sensors using either our custom script or standard ROS tools:

1. **Using the custom extraction script:**

  ```bash
  python3 extract_transforms.py <path_to_rosbag> <base_frame> <child_frame> --format matrix
  ```

  Example:
  ```bash
  python3 extract_transforms.py <path_to_rosbag> livox_imu livox_lidar --format matrix
  ```

  Note: Ensure the bag you provide has the `/tf_static` topic (such as *tf_model.bag or *tf_minimal.bag in the missions).

2. **Using ROS1 tf_echo tool:**

  ```bash
  rosrun tf tf_echo <reference_frame> <target_frame>
  ```

  <details className="rounded-md border border-gray-100 bg-gray-50 p-2">
    <summary className="cursor-pointer select-none text-sm font-medium pt-[20px]">
     <span className="inline-flex items-center">
      Example — reading transform from TF (ROS 1)
     </span>
    </summary>
    <div className="mt-3">
     The easiest way to retrieve the transform between two frames using ROS 1 is the <a href="http://wiki.ros.org/tf/Debugging%20tools" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-red-500 underline font-medium">tf_echo</a> functionality.<br/><br/>
     To get the transformation from <code>&lt;reference_frame&gt;</code> to <code>&lt;target_frame&gt;</code>, type:

    rosrun tf tf_echo reference_frame target_frame


  <strong>Note:</strong> <code>tf_echo</code> by default truncates numerical values in its output. See the <a href="https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf_echo.cpp#L123" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-red-500 underline font-medium">tf_echo implementation</a> for details.
  </details>

### Understanding Transform Outputs

<strong>Interpretation:</strong>
<br />
The transform output represents the rigid body transform (translation and rotation) from the <code>&lt;reference_frame&gt;</code> to the <code>&lt;target_frame&gt;</code>, as specified in [REP 105](https://www.ros.org/reps/rep-0105.html).

The output represents the pose of <code>&lt;target_frame&gt;</code> <strong>expressed in the coordinate system of</strong> <code>&lt;reference_frame&gt;</code>:
<ul>
  <li>
   The translation vector gives the position of the <code>&lt;target_frame&gt;</code> origin relative to <code>&lt;reference_frame&gt;</code> (in meters).
  </li>
  <li>
   The rotation (unit quaternion, in <code>[x, y, z, w]</code> order) gives the orientation of <code>&lt;target_frame&gt;</code> relative to <code>&lt;reference_frame&gt;</code>.
  </li>
</ul>

To transform a point from <code>&lt;reference_frame&gt;</code> to <code>&lt;target_frame&gt;</code>:
<div className="mb-2 mt-2" style={{ display: 'flex', justifyContent: 'center' }}>
  <code>
   p<sub>target_frame</sub> = R &middot; p<sub>reference_frame</sub> + t
  </code>
</div>
where <code>R</code> is the rotation matrix and <code>t</code> is the translation vector. The conventions for these transforms, including axis orientation and handedness, follow [REP 105](https://www.ros.org/reps/rep-0105.html).

### 2-) Depth and Confidence Image extraction

*Documentation coming soon*

## Launch Examples

*Coming soon*
