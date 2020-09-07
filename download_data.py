import os, argparse
from download_gdrive import *
from misc import dataset_split, extract

parser = argparse.ArgumentParser()

parser.add_argument('--link',         required=True,  type=str,  help='gdrive link')
parser.add_argument('--file_name',    required=True,  type=str,  help='file name')
parser.add_argument('--ext_and_part', required=False, type=bool, default=False, help='extract and partition operation needed or not')

opt = parser.parse_args()

# download compressed file
download_file_from_google_drive(opt.link, opt.file_name)

if opt.ext_and_part:
    # extract compressed file
    extract(opt.file_name, opt.file_name[: opt.file_name.index('.')])

    # data partitioning
    dataset_split(opt.file_name[: opt.file_name.index('.')])
