import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import test_restnet
import test_VGG
import os
from gradcam_vis import visualize_cam, Normalize, GradCAM, GradCAMpp
import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image


def res():
    net = test_VGG.NN()
    cam_dict = dict()
    vgg_model_dict = dict(type='vgg', arch=net, layer_name='layer5')
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]



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

    pil_img = im_orig.convert('RGB')
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = F.interpolate(torch_img, size=(32,32), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

    images = []
    for gradcam, gradcam_pp in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask.cpu(), torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

    images = make_grid(torch.cat(images, 0), nrow=5)

    output_dir = 'outputs'
    img_name = 'l5.jpeg'

    os.makedirs(output_dir, exist_ok=True)
    output_name = img_name
    output_path = os.path.join(output_dir, output_name)

    save_image(images, output_path)
    PIL.Image.open(output_path)


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
