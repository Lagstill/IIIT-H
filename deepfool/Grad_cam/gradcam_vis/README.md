## Grad-CAM: Gradient-weighted Class Activation Mapping
### Abstract


We propose a technique for making Convolutional Neural Network (CNN)-based models more transparent by visualizing the regions of input that are "important" for predictions from these models - or visual explanations. Our approach, called Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient information flowing into the final convolutional layer of a CNN to produce a coarse localization map of the important regions in the image. Grad-CAM is a strict generalization of the Class Activation Mapping. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to any CNN-based architectures. We also show how Grad-CAM may be combined with existing pixel-space visualizations to create a high-resolution class-discriminative visualization (Guided Grad-CAM). We generate Grad-CAM and Guided Grad-CAM visual explanations to better understand image classification, image captioning, and visual question answering (VQA) models. In the context of image classification models, our visualizations (a) lend insight into their failure modes showing that seemingly unreasonable predictions have reasonable explanations, and (b) outperform pixel-space gradient visualizations (Guided Backpropagation and Deconvolution) on the ILSVRC-15 weakly supervised localization task. For image captioning and VQA, our visualizations expose the somewhat surprising insight that common CNN + LSTM models can often be good at localizing discriminative input image regions despite not being trained on grounded image-text pairs. Finally, we design and conduct human studies to measure if Guided Grad-CAM explanations help users establish trust in the predictions made by deep networks. Interestingly, we show that Guided Grad-CAM helps untrained users successfully discern a "stronger" deep network from a "weaker" one even when both networks make identical predictions.



![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Grad_cam/gradcam_vis/gradcamm.png?raw=true)