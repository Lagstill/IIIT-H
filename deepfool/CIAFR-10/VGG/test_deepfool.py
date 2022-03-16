import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import test_restnet
import test_VGG
import os


def res():
    net = test_VGG.NN()

    # Switch to evaluation mode
    # net.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    im_orig = Image.open('ship.png')
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Remove the mean
    # im = transforms.Compose([
    #     #transforms.Scale(256),
    #     transforms.CenterCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean,
    #                          std=std)])(im_orig)
   # p=im_orig.unsqueeze(0)
   # print((im.unsqueeze(0)).size)
    r, loop_i, label_orig, label_pert, pert_image = deepfool(im_orig, net)

    # labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    #
    # str_label_orig = labels[int(label_orig)].split(',')[0]
    # str_label_pert = labels[int(label_pert)].split(',')[0]

    # print("Original label = ", str_label_orig)
    # print("Perturbed label = ", str_label_pert)

    plt.figure()
    plt.imshow(im_orig)
    plt.title(classes[label_orig])
    plt.show()

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 513)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                             transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(512)])

    plt.figure()
    plt.imshow(tf(pert_image.cpu()[0]))
    plt.title(classes[label_pert])
    plt.show()


if __name__ == '__main__':
    res()
