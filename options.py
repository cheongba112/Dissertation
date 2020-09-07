import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', required=False, type=int, default=16,          help='batch size')
parser.add_argument('--dataroot',   required=False, type=str, default='./UTKFace', help='path to dataset')
parser.add_argument('--epoch_num',  required=False, type=int, default=50,          help='epoch number')

# model paths for testing
parser.add_argument('--netE_path',  required=False, type=str, help='path to pre-trained encoder model')
parser.add_argument('--netG_path',  required=False, type=str, help='path to pre-trained generator model')
parser.add_argument('--netR_path',  required=False, type=str, help='path to pre-trained age regressor model')

opt = parser.parse_args()
