# download_datasets.py
# NOTE: This script includes pointers to where to download datasets.
# For Camelyon16 you may need to manually download or use available mirrors.
import os
import argparse

def main(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print("Please download BreakHis/BACH/Camelyon16 datasets manually or set up links.\n")
    print("BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
    print("BACH: https://www.icpr2020.org/challenge-bach/")
    print("Camelyon16: https://camelyon16.grand-challenge.org/")
    print("This script creates folders but you must follow dataset terms and download per their sites.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw', help='where to store')
    args = parser.parse_args()
    main(args.out) 