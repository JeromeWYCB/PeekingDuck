# Privacy Protection (Faces)

## Overview

As organisations collect more data, there is a need to better protect the identities of individuals in public and private places. AI Singapore has developed a solution that performs face anonymisation. This can be used to comply with the General Data Protection Regulation (GDPR) or other data privacy laws.

Our solution automatically detects and mosaics (or blurs) human faces. This is further elaborated in a [subsequent section](#how-it-works).

## Demo

To try our solution on your own computer with [PeekingDuck installed](../getting_started/01_installation.md): use the following configuration file: [privacy_protection_faces.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_protection_faces.yml) and run PeekingDuck.

```
> peekingduck run --config_path <path_to_privacy_protection_faces.yml>
```

## How it Works

There are two main components to face anonymisation: 1) face detection using AI; and 2) face de-identification. 

**1. Face Detection**

We use an open source face detection model known as [MTCNN](https://arxiv.org/abs/1604.02878) to identify human faces. This allows the application to identify the location of human faces in a video feed. The location is returned as two (x, y) coordinates in the form [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right. These are used to form the bounding box of each human face detected. For more information on how adjust the MTCNN node, checkout the [MTCNN configurable parameters](/peekingduck.pipeline.nodes.model.mtcnn.Node).

We can also use the [Yolo face model](/peekingduck.pipeline.nodes.model.yolo_face.Node) as an alternative. The yolo face model is a two-class model capable of differentiating human faces with and without face masks.

**2. Face De-Identification**

To perform face de-identification, we pixelate or gaussian blur the areas bounded by the bounding boxes.

## Nodes Used

These are the nodes used in the earlier demo (also in [privacy_protection_faces.yml](https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/privacy_protection_faces.yml)):
```
nodes:
- input.live
- model.mtcnn
- dabble.fps
- draw.mosaic_bbox
- draw.legend
- output.screen
```

**1. Face Detection Node**

As mentioned, we use the MTCNN model for face detection. You can also try the [Yolo face model](/peekingduck.pipeline.nodes.model.yolo_face.Node) that is included in our repo. By default, it uses the Yolov4-tiny model to detect faces. For better accuracy, you can change the parameters in the run config declaration to use the Yolov4 model instead.

**2. Face De-Identification Nodes**

You can mosaic or blur the faces detected using the`draw.mosaic_bbox` or `draw.blur_bbox` in the run config declaration.

**3. Adjusting Nodes**

With regard to the MTCNN model, some common node behaviours that you might want to adjust are:
- `mtcnn_min_size`: This specifies the minimum height and width of a face to be detected (default = 40 pixels). You may want to decrease the minimum size to increase the number of detections.
- `mtcnn_thresholds`: This specifies the threshold values for the Proposal Network (P-Net), Refine Network (R-Net) and Output Network (O-Net) in the MTCNN model. Calibration is performed at each stage in which bounding boxes with confidence scores less than the specified threshold (default = [0.6, 0.7, 0.7]) are discarded. 
- `mtcnn_score`: This specifies the threshold value in the final output. Bounding boxes with confidence scores less than the specified threshold (default = 0.7) in the final output are discarded. You may want to lower the mtcnn_thresholds and the mtcnn_score to increase the number of detections.

In addition, some common node behaviours that you might want to adjust for the mosaic_bbox and blur_bbox nodes are:
- `mosaic_level`: This defines the resolution of a mosaic filter (width x height). The number corresponds to the number of rows and columns used to create a mosaic. For example, the default setting (mosaic_level: 7) creates a 7 x 7 mosaic filter. Increasing the number increases the intensity of pixelation over an area.
- `blur_level`: TBC.