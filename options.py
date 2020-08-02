import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', required=False, type=int, default=64, help='batch size')
parser.add_argument('--dataroot', required=False, type=str, default='./14', help='path to dataset')

opt = parser.parse_args()
