import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import yaml
import os
import torch
import numpy as np

from src.models import ResNet50Fine, ViTModel
from src.xai import XAIProcessor, overlay_heatmap

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class XAIApp:
    def __init__(self, master, cfg):
        self.master = master
        self.cfg = cfg
        self.device = get_device()

        master.title('Breast Histopathology XAI Tool')
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(fill='both', expand=True)

        self.img_panel = tk.Label(self.top_frame)
        self.img_panel.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.left_panel = tk.Label(self.top_frame)
        self.left_panel.grid(row=1, column=0, padx=10, pady=5)
        self.right_panel = tk.Label(self.top_frame)
        self.right_panel.grid(row=1, column=1, padx=10, pady=5)

        self.controls = tk.Frame(master)
        self.controls.pack(fill='x', padx=10, pady=8)

        self.load_btn = tk.Button(self.controls, text='Load Image', command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=6)

        self.run_btn = tk.Button(self.controls, text='Run XAI', command=self.run_xai_thread)
        self.run_btn.grid(row=0, column=1, padx=6)

        self.export_btn = tk.Button(self.controls, text='Export Overlays', command=self.export_overlays)
        self.export_btn.grid(row=0, column=2, padx=6)

        self.opacity_scale = tk.Scale(self.controls, from_=0, to=100, orient='horizontal', label='Overlay Opacity')
        self.opacity_scale.set(60)
        self.opacity_scale.grid(row=0, column=3, padx=6)

        self.status_label = tk.Label(master, text='Status: Ready')
        self.status_label.pack(side='bottom', fill='x')

        
        self.model = None
        self.xai = None
        self._load_model()

        
        self.img = None
        self.grad_overlay = None
        self.shap_overlay = None

    def _load_model(self):
        
        model_type = self.cfg['model']['type']
        try:
            if model_type == 'resnet':
                self.model = ResNet50Fine(num_classes=self.cfg['model']['num_classes'])
            else:
                self.model = ViTModel(model_name=self.cfg['model'].get('vit_name','vit_base_patch16_224'),
                                      num_classes=self.cfg['model']['num_classes'])
            checkpoint_path = os.path.join(self.cfg['training']['outdir'], f'best_{model_type}.pt')
            if not os.path.exists(checkpoint_path):
                self.status_label.config(text=f'Status: Warning - checkpoint not found at {checkpoint_path}')
            else:
                state = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.xai = XAIProcessor(self.model, self.device, model_type=model_type)
            self.status_label.config(text='Status: Model loaded')
        except Exception as e:
            messagebox.showerror('Model load error', str(e))
            self.status_label.config(text='Status: Model load failed')

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('PNG Image','*.png'),('JPEG','*.jpg;*.jpeg'),('All files','*.*')])
        if not path:
            return
        self.img_path = path
        self.img = Image.open(path).convert('RGB')
        disp = self.img.resize((512,512))
        self.img_tk = ImageTk.PhotoImage(disp)
        self.img_panel.config(image=self.img_tk)
        self.status_label.config(text=f'Status: Loaded {os.path.basename(path)}')

    def run_xai_thread(self):
        if self.img is None:
            messagebox.showinfo('No image', 'Please load an image first.')
            return
        threading.Thread(target=self.compute_xai).start()

    def compute_xai(self):
        self.status_label.config(text='Status: Computing XAI...')
        try:
            
            grad_map = None
            shap_map = None
            try:
                grad_map = self.xai.gradcam(self.img, upsample_size=(224,224))
            except Exception as e:
                print('Grad-CAM failed:', e)
                grad_map = np.zeros((224,224), dtype=np.float32)

            bg_dir = self.cfg['data'].get('bg_dir', None)
            background_images = []
            if bg_dir and os.path.isdir(bg_dir):
            
                import random
                fls = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
                random.shuffle(fls)
                for f in fls[:5]:
                    try:
                        background_images.append(Image.open(os.path.join(bg_dir,f)).convert('RGB'))
                    except Exception:
                        pass
           
            if len(background_images) == 0:
                background_images = [self.img]

            try:
                shap_map = self.xai.shap_gradient_explainer(self.img, background_images, nsamples=50)
            except Exception as e:
                print('SHAP failed:', e)
                shap_map = self.xai.shap_superpixel(self.img, n_segments=80, nsamples=120)

            alpha = float(self.opacity_scale.get()) / 100.0
            self.grad_overlay = overlay_heatmap(self.img, grad_map, alpha=alpha)
            self.shap_overlay = overlay_heatmap(self.img, shap_map, alpha=alpha)

            left_thumb = self.grad_overlay.resize((384,384))
            right_thumb = self.shap_overlay.resize((384,384))
            self.left_tk = ImageTk.PhotoImage(left_thumb)
            self.right_tk = ImageTk.PhotoImage(right_thumb)
            self.left_panel.config(image=self.left_tk)
            self.right_panel.config(image=self.right_tk)
            self.status_label.config(text='Status: XAI complete')
        except Exception as e:
            self.status_label.config(text='Status: XAI failed')
            print('Compute XAI error:', e)

    def export_overlays(self):
        if self.grad_overlay is None or self.shap_overlay is None:
            messagebox.showinfo('Nothing to export', 'Run XAI first.')
            return
        out_dir = filedialog.askdirectory()
        if not out_dir:
            return
        stem = os.path.splitext(os.path.basename(self.img_path))[0]
        grad_path = os.path.join(out_dir, stem + '_gradcam.png')
        shap_path = os.path.join(out_dir, stem + '_shap.png')
        self.grad_overlay.save(grad_path)
        self.shap_overlay.save(shap_path)
        messagebox.showinfo('Exported', f'Exported:\n{grad_path}\n{shap_path}')
        self.status_label.config(text=f'Status: Exported overlays to {out_dir}')

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml'))
    root = tk.Tk()
    app = XAIApp(root, cfg)
    root.mainloop()