# Einstein_View
### Running the code
Run the code from main.py. You would need to ensure all depedencies for the models mentioned below are met.

Please note: We have included a separate video for collision detection as we did not get that feature working earlier. (Please run einstein.py for that)

### Models Used

ultralytics/YOLOWorld: We use "yolov8s-worldv2.pt" for detecting vehicles, traffic lights, stop signs and pedestrians. 

ZoeDepth: We use "Intel/zoedepth-kitti" for estimating relative depth of detected objects in the scene.

MaskRCNN: We use MASKRCNN (https://github.com/matterport/Mask_RCNN) for detecting and classifying lanes in the scene.

Detic: We used Detic model for many subcategories like dustbin , cones , barrels etc . 

Florence-2: We used this model to get text on the board for road signs. Filtered boards for speed limit and rendered the speed board in blender. We also use it for traffic light type and direction detection (in case of arrow).

OSX: We use OSX for human pose detection and rendering.

Openpifpaf: We use Openpifpaf over it to check if at the given pose , the key features corresponding to that pose are visible  and correct the poses from YOLO3D accordingly.
