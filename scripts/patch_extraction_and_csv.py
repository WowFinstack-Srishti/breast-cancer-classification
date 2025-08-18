#!/usr/bin/env python3
import os
import h5py
import random
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

from stain_norm import normalize_staining  # keep stain_norm.py in src/

def ensure_rgb(img):
    """Ensure image is RGB numpy array."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 2:  # grayscale
        img = np.stack([img]*3, axis=-1)
    return img

def extract_pcam(h5_path, label_path, out_dir, split, img_per_file=None):
    with h5py.File(h5_path, 'r') as hf_x, h5py.File(label_path, 'r') as hf_y:
        X = hf_x['x'][:]  # shape: (N, 96,96,3)
        y = hf_y['y'][:]  # shape: (N,)
    indices = list(range(len(y)))
    if img_per_file:
        indices = random.sample(indices, img_per_file)
    os.makedirs(out_dir, exist_ok=True)
    csv = []
    for i in indices:
        arr = X[i]  # uint8
        label = int(y[i])
        arr = normalize_staining(ensure_rgb(arr))  # stain norm
        fname = f'{split}_{i:06d}.png'
        path = os.path.join(out_dir, fname)
        Image.fromarray(arr).save(path)
        csv.append({'filename': f'pcam/{split}/{fname}', 'label': label})
    return csv

def extract_breakhis(raw_dir, out_dir, magnifications=None):
    samples = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(('.png','.tif','.jpg')):
                parts = f.split('-')
                if len(parts) >= 5:
                    mag = parts[-2]
                    if (magnifications and mag not in magnifications):
                        continue
                    label = 1 if parts[1] == 'M' else 0  # Malignant=1, Benign=0
                    filepath = os.path.join(root, f)
                    samples.append((filepath, label))
    random.shuffle(samples)
    os.makedirs(out_dir, exist_ok=True)
    csv = []
    for idx, (fp, label) in enumerate(samples):
        img = Image.open(fp).convert('RGB').resize((224,224))
        img = normalize_staining(ensure_rgb(img))
        fn = f'breakhis_{idx:05d}.png'
        Image.fromarray(img).save(os.path.join(out_dir, fn))
        csv.append({'filename': f'breakhis/{fn}', 'label': label})
    return csv

def extract_bach(raw_dir, out_dir):
    samples = []
    for cls in os.listdir(raw_dir):
        d = os.path.join(raw_dir, cls)
        if os.path.isdir(d):
            label = {'normal':0,'benign':1,'in_situ':2,'invasive':3}.get(cls.lower(), None)
            if label is None: continue
            for f in os.listdir(d):
                if f.lower().endswith(('.png','.tif','.jpg')):
                    samples.append((os.path.join(d,f), label))
    random.shuffle(samples)
    os.makedirs(out_dir, exist_ok=True)
    csv = []
    for idx,(fp,label) in enumerate(samples):
        img = Image.open(fp).convert('RGB')
        w,h = img.size
        step = 224
        count = 0
        for y in range(0, h - step +1, step):
            for x in range(0, w - step +1, step):
                patch = img.crop((x,y,x+step,y+step))
                patch = normalize_staining(ensure_rgb(patch))
                fn = f'bach_{idx:04d}_{count:03d}.png'
                Image.fromarray(patch).save(os.path.join(out_dir, fn))
                csv.append({'filename': f'bach/{fn}', 'label': label})
                count+=1
    return csv

def split_and_save(csv_list, processed_csv_dir):
    df = pd.DataFrame(csv_list)
    train, rest = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest['label'], random_state=42)
    os.makedirs(processed_csv_dir, exist_ok=True)
    train.to_csv(os.path.join(processed_csv_dir,'train.csv'), index=False)
    val.to_csv(os.path.join(processed_csv_dir,'val.csv'), index=False)
    test.to_csv(os.path.join(processed_csv_dir,'test.csv'), index=False)
    print("✅ CSV splits created at", processed_csv_dir)

def main(args):
    all_csv = []
    os.makedirs(args.out_patches, exist_ok=True)
    # PCam
    if args.pcam:
        for split in ['train','valid','test']:
            x_h5 = os.path.join(args.pcam, f'camelyonpatch_level_2_split_{split}_x.h5')
            y_h5 = os.path.join(args.pcam, f'camelyonpatch_level_2_split_{split}_y.h5')
            out = os.path.join(args.out_patches, 'pcam', split)
            all_csv += extract_pcam(x_h5, y_h5, out, split, img_per_file=None)
    # BreakHis
    if args.breakhis:
        out = os.path.join(args.out_patches, 'breakhis')
        all_csv += extract_breakhis(args.breakhis, out, magnifications=args.magnifications)
    # BACH
    if args.bach:
        out = os.path.join(args.out_patches, 'bach')
        all_csv += extract_bach(args.bach, out)
    split_and_save(all_csv, args.csv_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcam', help='PCam raw folder')
    parser.add_argument('--breakhis', help='BreakHis raw folder')
    parser.add_argument('--bach', help='BACH raw folder')
    parser.add_argument('--out_patches', default='data/processed/patches')
    parser.add_argument('--csv_dir', default='data/processed')
    parser.add_argument('--magnifications', nargs='*', default=None, help='BreakHis magnifications to include e.g. 40 100')
    args = parser.parse_args()
    main(args)
