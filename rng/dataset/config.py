import logging

import numpy as np
import open3d as o3d

camera = "realsense"
logging.warn(f"Using {camera} camera now!")


def get_camera_intrinsic(camera=camera):
    if camera == "kinect":
        intrinsics = np.array(
            [[631.55, 0, 638.43], [0, 631.21, 366.50], [0, 0, 1]], dtype=np.float32
        )
    elif camera == "realsense":
        intrinsics = np.array(
            [
                [538.17646753, 0, 321.72223033],
                [0, 538.23831376, 238.63675485],
                [0, 0, 1.0],
            ]
        )
    else:
        raise ValueError('Camera format must be either "kinect" or "realsense".')
    return intrinsics


# # realsense
# intrinsics = np.array([[927.16973877, 0, 651.31506348],
#                        [0, 927.36688232, 349.62133789], [0, 0, 1]])

# # kinect
# intrinsics = np.array([[631.55, 0, 638.43],
#                        [0, 631.21, 366.50], [0, 0, 1]])
