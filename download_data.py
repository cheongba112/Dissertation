import argparse
from download_gdrive import *

parser = argparse.ArgumentParser()

parser.add_argument('--link',         required=True,  type=str,  help='gdrive link')
parser.add_argument('--file_name',    required=True,  type=str,  help='file name')

opt = parser.parse_args()

# download compressed file
download_file_from_google_drive(opt.link, opt.file_name)
