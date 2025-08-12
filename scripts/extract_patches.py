import os
import openslide
from PIL import Image
import numpy as np

def tile_wsi(wsi_path, out_dir, patch_size=224, level=0, stride=224, bg_thresh=0.8):
    os.makedirs(out_dir, exist_ok=True)
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.level_dimensions[level]
    count=0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            arr = np.array(patch)
            white_frac= np.mean(np.all(arr>240, axis=2))
            if white_frac > bg_thresh:
                continue
            out_path = os.path.join(out_dir,f'{os.path.basename(wsi_path)}_{x}_{y}.png')
            patch.save(out_path)
            count += 1
            print(f'tiled {count} patches')

if __name__ == '__main__':
    import sys
    wsi = sys.argv[1]
    out = sys.argv[2]
    tile_wsi(wsi, out)

