import os, shutil, random, tarfile


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


# split test set and valid set
def dataset_split(root_path):
    old_root = root_path
    test_root = root_path + '_test'
    valid_root = root_path + '_valid'

    if not os.path.exists(test_root):
        os.mkdir(test_root)

    if not os.path.exists(valid_root):
        os.mkdir(valid_root)

    files = os.listdir(old_root)

    tbmoved = random.sample(range(len(files)), 100)
    for i in tbmoved:
        shutil.move(os.path.join(old_root, files[i]), os.path.join(test_root, files[i]))

    tbmoved = random.sample(range(len(files)), 8)
    for i in tbmoved:
        shutil.move(os.path.join(old_root, files[i]), os.path.join(valid_root, files[i]))


# extract compressed file
def extract(tar_path, target_path):

    if tar_path[tar_path.index('.'): ] == '.tar':
        tar = tarfile.open(tar_path, "r")
    elif tar_path[tar_path.index('.'): ] == '.tar.gz':
        tar = tarfile.open(tar_path, "r:gz")

    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, target_path)
    tar.close()
