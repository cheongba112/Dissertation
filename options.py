import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', required=False, type=int, default=32,            help='batch size')
parser.add_argument('--dataroot',   required=False, type=str, default='./cacd_lite', help='path to dataset')
parser.add_argument('--epoch_num',  required=False, type=int, default=20,            help='epoch number')

opt = parser.parse_args()
