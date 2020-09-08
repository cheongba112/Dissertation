import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str)
opt = parser.parse_args()


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


if __name__ == '__main__':
    dataset_split(opt.path)
