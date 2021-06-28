from typing import List, Tuple, Any, Iterable, Union
import numpy as np
import cv2
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA
from peekingduck.pipeline.nodes.draw.utils.constants import \
    CHAMPAGNE, THICK
from peekingduck.pipeline.nodes.draw.utils.general import \
    get_image_size, project_points_onto_original_image

SKELETON_SHORT_NAMES = (
    "N", "LEY", "REY", "LEA", "REA", "LSH",
    "RSH", "LEL", "REL", "LWR", "RWR",
    "LHI", "RHI", "LKN", "RKN", "LAN", "RAN")

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
            [3, 5], [4, 6], [5, 7]]

KEYPONT_DOT_RADIUS = 7

def draw_human_poses(image: np.array,
                     keypoints: np.ndarray,
                     keypoint_scores: np.ndarray,
                     keypoint_conns: np.ndarray) -> None:
    # pylint: disable=too-many-arguments
    """Draw poses onto an image frame.

    Args:
        image (np.array): image of current frame
        keypoints (List[Any]): list of keypoint coordinates
        keypoints_scores (List[Any]): list of keypoint scores
        keypoints_conns (List[Any]): list of keypoint connections
    """
    image_size = get_image_size(image)
    num_persons = keypoints.shape[0]
    if num_persons > 0:
        for i in range(num_persons):
            _draw_connections(image, keypoint_conns[i],
                              image_size, CHAMPAGNE)
            _draw_keypoints(image, keypoints[i],
                            keypoint_scores[i], image_size,
                            CHAMPAGNE, KEYPONT_DOT_RADIUS)


def _draw_connections(frame: np.array,
                      connections: Union[None, Iterable[Any]],
                      image_size: Tuple[int, int],
                      connection_color: Tuple[int, int, int]) -> None:
    """ Draw connections between detected keypoints """
    if connections is not None:
        for connection in connections:
            pt1, pt2 = project_points_onto_original_image(connection, image_size)
            cv2.line(frame,
                     (pt1[0], pt1[1]),
                     (pt2[0], pt2[1]),
                     connection_color,
                     THICK['thickness'])


def _draw_keypoints(frame: np.ndarray,
                    keypoints: np.ndarray,
                    scores: np.ndarray,
                    image_size: Tuple[int, int],
                    keypoint_dot_color: Tuple[int, int, int],
                    keypoint_dot_radius: int) -> None:
    # pylint: disable=too-many-arguments
    """ Draw detected keypoints """
    img_keypoints = project_points_onto_original_image(
        keypoints, image_size)

    for idx, keypoint in enumerate(img_keypoints):
        _draw_one_keypoint_dot(frame, keypoint, keypoint_dot_color, keypoint_dot_radius)


def _draw_one_keypoint_dot(frame: np.ndarray,
                           keypoint: np.ndarray,
                           keypoint_dot_color: Tuple[int, int, int],
                           keypoint_dot_radius: int) -> None:
    """ Draw single keypoint """
    cv2.circle(frame, (keypoint[0], keypoint[1]), keypoint_dot_radius, keypoint_dot_color, -1)
