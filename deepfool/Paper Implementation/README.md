# DeepFool
DeepFool is a simple algorithm to find the minimum adversarial perturbations in deep networks

State-of-the-art deep neural networks have achieved impressive results on many image classification tasks. However, these same architectures have been shown to be unstable to small, well sought, perturbations of the images. Despite the importance of this phenomenon, no effective methods have been proposed to accurately compute the robustness of state-of-the-art deep classifiers to such perturbations on large-scale datasets. In this paper, we fill this gap and propose the DeepFool algorithm to efficiently compute perturbations that fool deep networks, and thus reliably quantify the robustness of these classifiers. Extensive experimental results show that our approach outperforms recent methods in the task of computing adversarial perturbations and making classifiers more robust.

### deepfool.py

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) using PyTorch to find adversarial perturbations.

__Note__: The final softmax (loss) layer should be removed in order to prevent numerical instabilities.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `num_classes`: limits the number of classes to test against, by default = 10.
- `max_iter`: max number of iterations, by default = 50.

### test_deepfool.py

A simple demo which computes the adversarial perturbation for a test image from ImageNet dataset.

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.

## Inferences

From the implementation work of https://github.com/LTS4/DeepFool/tree/master/Python , using a pretraind resnet model from the ImageNet dataset the sample image was deepfooled by
the algorithm as proposed and was successsfully perturbed though, from my conclusion the image instead of being perturbed was zoomed as follows:

### ORIGINAL IMAGE :
![Original label =  n03538406 horse cart](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Paper%20Implementation/original.png?raw=true)

### PERTURBED IMAGE :
![Perturbed label =  n03967562 plow](https://github.com/Lagstill/IIIT-H/blob/main/deepfool/Paper%20Implementation/perturbed.png?raw=true)
