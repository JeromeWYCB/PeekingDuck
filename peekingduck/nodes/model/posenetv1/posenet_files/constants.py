# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Constants used for PoseNet
"""

import numpy as np

KEYPOINT_NAMES = (
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
)

_KEYPOINT_IDS = {pn: pid for pid, pn in enumerate(KEYPOINT_NAMES)}

_POSE_CHAIN = [
    ("nose", "leftEye"),
    ("leftEye", "leftEar"),
    ("nose", "rightEye"),
    ("rightEye", "rightEar"),
    ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
]

POSE_CONNECTIONS = np.array(
    [(_KEYPOINT_IDS[parent], _KEYPOINT_IDS[child]) for parent, child in _POSE_CHAIN]
)

IMAGE_NET_MEAN = [-123.15, -115.90, -103.06]
LOCAL_MAXIMUM_RADIUS = 1
MIN_PART_SCORE = 0.1
MIN_ROOT_SCORE = 0.5
NMS_RADIUS = 20
OUTPUT_STRIDE = 16
