import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image
from deepfool import deepfool
import test_restnet


def res():
    net = test_restnet.NN()

    im_orig = Image.open('ship.png')
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)])(im_orig)
    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

    plt.figure()
    plt.imshow(im_orig)
    plt.title(classes[label_orig])
    plt.show()

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                             transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])

    plt.figure()
    plt.imshow(tf(pert_image.cpu()[0]))
    plt.title(classes[label_pert])
    plt.show()


if __name__ == '__main__':
    res()
