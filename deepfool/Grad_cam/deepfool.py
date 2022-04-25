import numpy
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import collections.abc as container_abcs
import torchvision.transforms as transforms


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # im_orig = Image.open('ship.png')
    im = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])(image)
    f_image = net.forward(im.unsqueeze(0)).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    pert_image = copy.deepcopy(im.unsqueeze(0))
    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = net(x)
    k_i = label
    r_tot = np.zeros(pert_image.shape)
    w = np.zeros(pert_image.shape)

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        pert = torch.from_numpy(numpy.array(pert, dtype=np.float32))
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_i = r_i.detach().numpy()
        r_tot = np.float32(r_tot + r_i)
        print("R TOT:", type(torch.from_numpy(r_tot)), "IMG : ", type(image))
        if is_cuda:
            pert_image = im + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = im + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
