# tensor to string of list
def tl(tensor):
    return str(tensor.tolist())


# weights initialise function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


# TV loss
def TV_Loss(imgTensor, img_size=128):
    x = (imgTensor[:, :, 1:, :] - imgTensor[:, :, :img_size - 1, :]) ** 2
    y = (imgTensor[:, :, :, 1:] - imgTensor[:, :, :, :img_size - 1]) ** 2

    out = (x.mean(dim=2) + y.mean(dim=3)).mean()
    return out
