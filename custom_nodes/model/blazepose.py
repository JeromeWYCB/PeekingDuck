# Copyright 2021 AI Singapore
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

from typing import Any, Dict

import cv2
import numpy as np
import mediapipe as mp

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode = False,
                                      model_complexity = 1,
                                      min_detection_confidence=0.7,
                                      min_tracking_confidence=0.7)             

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Detects 33 pose landmarks or keypoints using the BlazePose GHUM 3D model.

        Args:
           inputs (dict): Dict with key "img".

        Returns:
           outputs (dict): A dictionary of np.ndarrays i.e. keypoints, keypoint_depth, 
           keypoint_scores.
        """         
        image = cv2.cvtColor(inputs['img'], cv2.COLOR_BGR2RGB)
        image.flags.writeable = False 
        results = self.pose.process(inputs['img'])

        image= cv2.cvtColor(inputs['img'], cv2.COLOR_RGB2BGR)
        image.flags.writeable = True 

        self.mp_drawing.draw_landmarks(inputs['img'], 
                                       results.pose_landmarks, 
                                       self.mp_pose.POSE_CONNECTIONS)

        outputs = self.get_17keypoints(self.get_output(results.pose_landmarks))                                 

        return {'keypoints': outputs['keypoints'], 
                'keypoint_depth': outputs['keypoint_depth'], 
                'keypoint_scores': outputs['keypoint_scores']}

    @staticmethod
    def get_output(pose_landmarks):
        """Helper function to obtain and format the outputs from the BlazePose 
        GHUM 3D model.

        Args:
            pose_landmarks (protobuf): x, y, z and visibility score per landmark.

        Returns:
            pose_outputs (dict): A dictionary of np.ndarrays i.e. keypoints, 
            keypoint_depth, keypoint_scores.
        """ 
        pose_outputs = {'keypoints': np.empty((0,2)),
                        'keypoint_depth': np.empty((0,1)),
                        'keypoint_scores': np.empty((0,1))}
        
        if pose_landmarks != None:
            for data_point in pose_landmarks.landmark:
                pose_outputs['keypoints'] = np.append(pose_outputs['keypoints'],
                                                      np.array([[data_point.x, data_point.y]]), 
                                                      axis = 0)        
                pose_outputs['keypoint_depth'] = np.append(pose_outputs['keypoint_depth'],
                                                           np.array([[data_point.z]]), 
                                                           axis = 0)        
                pose_outputs['keypoint_scores'] = np.append(pose_outputs['keypoint_scores'],
                                                            np.array([[data_point.visibility]]), 
                                                            axis = 0)
        else:
            pose_outputs['keypoints'] = np.append(pose_outputs['keypoints'],
                                                  np.zeros([33,2]), 
                                                  axis = 0)    
            pose_outputs['keypoint_depth'] = np.append(pose_outputs['keypoint_depth'],
                                                       np.zeros([33,1]), 
                                                       axis = 0)  
            pose_outputs['keypoint_scores'] = np.append(pose_outputs['keypoint_scores'],
                                                        np.zeros([33,1]), 
                                                        axis = 0)                                                            
        
        pose_outputs['keypoints'] = pose_outputs['keypoints'].reshape(1, 33, 2)
        pose_outputs['keypoint_depth'] = pose_outputs['keypoint_depth'].reshape(1, 33, 1)
        pose_outputs['keypoint_scores'] = pose_outputs['keypoint_scores'].reshape(1, 33, 1)

        return pose_outputs

    @staticmethod
    def get_17keypoints(pose_outputs):
        """Helper function to map and obtain 17 keypoints from the BlazePose 
        GHUM 3D model.

        Args:
            mp_output (dict): x, y, z and visibility score per landmark.

        Returns:
            outputs (dict): A dictionary of np.ndarrays i.e. keypoints, 
            keypoint_depth, keypoint_scores.
        """ 
        output = {'keypoints': np.empty((0,2)),
                  'keypoint_depth': np.empty((0,1)),
                  'keypoint_scores': np.empty((0,1))}
        
        keypoints = [0, 2, 5, 7, 8, 11, 12, 13 , 14 ,15 ,16 ,23 , 24, 25, 26, 27, 28]
        
        for keypoint in keypoints:
            output['keypoints'] = np.append(output['keypoints'],
                                            np.array([pose_outputs['keypoints']
                                                     [0][keypoint]]),
                                            axis = 0)
            output['keypoint_depth'] = np.append(output['keypoint_depth'],
                                                 np.array([pose_outputs['keypoint_depth']
                                                 [0][keypoint]]),
                                                 axis = 0)
            output['keypoint_scores'] = np.append(output['keypoint_scores'],
                                                  np.array([pose_outputs['keypoint_scores']
                                                           [0][keypoint]]),
                                                  axis = 0)
            
        output['keypoints'] = output['keypoints'].reshape(1, 17, 2)
        output['keypoint_depth'] = output['keypoint_depth'].reshape(1, 17, 1)
        output['keypoint_scores'] = output['keypoint_scores'].reshape(1, 17, 1)
        
        return output
