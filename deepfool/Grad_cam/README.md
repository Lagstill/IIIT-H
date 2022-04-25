## Grad-CAM: Visual Explanations from Deep Networks


Grad-CAM is a popular technique for visualizing where a convolutional neural network model is looking.

Grad-CAM is class-specific, meaning it can produce a separate visualization for every class present in the image.Grad-CAM can be used for weakly-supervised localization, i.e. determining the location of particular objects using a model that was trained only on whole-image labels rather than explicit location annotations.

Grad-CAM can also be used for weakly-supervised segmentation, in which the model predicts all of the pixels that belong to particular objects, without requiring pixel-level labels for training:

![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam//grad.png?raw=true)



### Outputs for the vgg CIFAR-10 deepfooled net

![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/l1.jpeg?raw=true)
![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/l2.jpeg?raw=true)
![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/l3.jpeg?raw=true)
![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/l4.jpeg?raw=true)
![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/l5.jpeg?raw=true)

