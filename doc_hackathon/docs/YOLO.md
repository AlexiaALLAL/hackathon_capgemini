# 1. YOLO implementation

You can find it in the `yolo/` folder.
We tried to run YOLO on our data but quickly stopped due to YOLO being trained with polygonal bounding boxes for segmentation. Since our data is only annotated with the masks of the classes on our images, it is not possible to directly feed these annotations to YOLO.

While trying to implement this code, we chose to give up on the temporal dimension, selecting only one image per sequence and choosing only the RBG channels in order to feed the images to YOLO.
