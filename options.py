import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', required=False, type=int, default=100, help='batch size')

opt = parser.parse_args()
