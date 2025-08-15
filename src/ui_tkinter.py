import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
import torch
from src.xai import XAIProcessor, transform
from src.models import ResNet50Fine, ViTModel

class XAIApp:
    def __init__(self, master, model, device, xai_processor):
        self.master = master
        self.model = model
        self.device = device
        self.xai = xai_processor
        master.title('Breast Histopathology XAI Tool')
        self.img_panel = tk.Label(master)
        self.img_panel.grid(row=0, column=0, columnspan=2)
        self.grad_panel = tk.Label(master)
        self.grad_panel.grid(row=1, column=0)
        self.shap_panel = tk.Label(master)
        self.shap_panel.grid(row=1, column=1)
        
        btn = tk.Button(master, text='Load Image', command=self.load_image)
        btn.grid(row=2, column=0)
        self.opacity = tk.Scale(master, from_=0, to=100, orient='horizontal', label='Overlay Opacity')
        self.opacity.set(60)
        self.opacity.grid(row=2, column=1)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.img = Image.open(path).convert('RGB')
        self.show_image(self.img)
        threading.Thread(target=self.compute_xai).start()

    def show_image(self, pil_img):
        disp = pil_img.resize((512,512))
        self.tkimg = ImageTk.PhotoImage(disp)
        self.img_panel.config(image=self.tkimg)

    def compute_xai(self):
        heat_grad = self.xai.gradcam(self.img)
        heat_shap = self.xai.shap_deepexplainer(self.img, [self.img])
        grad_overlay = self.make_overlay(self.img, heat_grad)
        shap_overlay = self.make_overlay(self.img, heat_shap)
        self.grad_tk = ImageTk.PhotoImage(grad_overlay.resize((256,256)))
        self.shap_tk = ImageTk.PhotoImage(shap_overlay.resize((256,256)))
        self.grad_panel.config(image=self.grad_tk)
        self.shap_panel.config(image=self.shap_tk)

    def make_overlay(self, img, heat):
        import cv2
        import numpy as np
        arr = np.array(img.resize((224,224)))
        h = (heat*255).astype(np.uint8)
        cmap = cv2.applyColorMap(h, cv2.COLORMAP_JET)
        cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        overlay = (0.6*cmap + 0.4*arr).astype(np.uint8)
        return Image.fromarray(overlay)

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('configs/default.yaml'))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ResNet50Fine(num_classes=2)
    model.load_state_dict(torch.load(cfg['training']['outdir'] + '/best_resnet.pt', map_location=device))
    model.to(device)
    xai = XAIProcessor(model, device)
    root = tk.Tk()
    app = XAIApp(root, model, device, xai)
    root.mainloop()