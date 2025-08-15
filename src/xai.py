import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop, IntegratedGradients
import shap
from skimage.segmentation import slic
from skimage.color import gray2rgb

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def to_numpy_img(pil_img, size=(224,224)):
    arr = np.array(pil_img.resize(size)).astype(np.uint8)
    return arr

class XAIProcessor:
    def __init__(self, model, device, target_layer=None, model_type='resnet'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.model_type = model_type
        if target_layer is None and model_type == 'resnet':
            try:
                self.target_layer = self.model.net.layer4[-1].conv3
            except Exception:
                raise ValueError('Could not find target layer for provided model. Supply target_layer.')
        else:
            self.target_layer = target_layer

    def _preprocess_pil(self, pil):
        return transform(pil).unsqueeze(0)
    
    def gradcam(self, pil_image, target_class=None, unsample_size=(224,224)):
        """Return normalized heatmap (H,W) in [0,1]"""
        inp = self._preprocess_pil(pil_image).to(self.device)
        if self.model_type == 'resnet' and self.target_layer is None:
            raise ValueError('target_layer required for gradcam on this model')
        lgc = LayerGradCam(self.model.net if hasattr(self.model,'net') else self.model, self.target_layer)
        with torch.no_grad():
            logits = self.model(inp.to(self.device))
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1). item())
        attr = lgc.attribute(inp.to(self.device), target=target_class)

        attr = LayerAttribution.interpolate(attr, upsample_size)
        heat = attr.squeeze().cpu().detach().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        return heat
    
    def guided_gradcam(self, pil_image, target_class=None):
        """Create a guided grad-cam overlay image"""

        inp = self._preprocess_pil(pil_image).to(self.device)
        gbp = GuidedBackprop(self.model.net if hasattr(self.model,'net') else self.model)
        if target_class is None:
            with torch.no_grad():
                logits = self.model(inp)
                target_class = int(torch.argmax(logits, dim=1).item())
        gb = gbp.attribute(inp, target=target_class)
        gb = gb.squeeze().cpu().detach().numpy()
        gb = np.transpose(gb, (1,2,0))
        gb = (gb - gb.min()) / (gb.max()-gb.min()+1e-9)
       
        gc = self.gradcam(pil_image, target_class)
        gc = np.expand_dims(gc, axis=2)
        guided = gb * gc
        guided = (guided - guided.min()) / (guided.max()-guided.min()+1e-9)
        return guided
    
    def shap_gradient_explainer(self, pil_image, background_images, nsamples=50):
        """Use shap.GradientExplainer or DeepExplainer depending on model compatibility. 
        background_images: list of PIL images (small set) 
        returns heatmap HxW normalized
        """
        try:
            def f(x):
                x_t = torch.tensor(x).permute(0,3,1,2).float().to(self.device)
                with torch.no_grad():
                    out = self.model(x_t)
                return out.cpu().numpy()
            bg_np = np.stack([to_numpy_img(im) for im in background_images])
            bg_np = bg_np.astype(np.float32)
            explainer = shap.GradientExplainer(f, bg_np[:min(len(bg_np),10)])
            inp_np = to_numpy_img(pil_image)[None].astype(np.float32)
            shap_vals = explainer.shap_values(inp_np, nsamples=nsamples)

            sv = np.sum(np.array(shap_vals), axis=-1)
            sv = sv[0]
            sv = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
            return sv
        except Exception as e:
            print('GradientExplainer failed, falling back to superpixel KernelSHAP:', e)
            return self.shap_superpixel(pil_image, n_segments=100, nsamples=200)

    def shap_superpixel(self, pil_image, n_segments=100, nsamples=200):
        """
        Superpixel KernelSHAP: segments image into superpixels and explains
        importance of each superpixel.
        Returns per-pixel heatmap (upsampled from segments)
        """
        img_np = to_numpy_img(pil_image)
        segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1)

        def masker(segments_mask, image, segments):
            out = image.copy()
            for seg_id in np.unique(segments):
                if segments_mask[seg_id] == 0:
                    out[segments==seg_id] = np.mean(image, axis=(0,1))
            return out
        
        def f(images):
            x = np.stack(images).astype(np.float32)
            x = torch.tensor(x).permute(0,3,1,2).float().to(self.device)
            with torch.no_grad():
                out = self.model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            return probs
        
        num_segs = segments.max()+1

        import random
        sampled_masks = []
        sampled_imgs = []
        for _ in range(nsamples):
            mask = np.random.choice([0,1], size=(num_segs,), p=[0.5,0.5])
            sampled_masks.append(mask)
            sampled_imgs.append(masker(mask, img_np, segments))
        probs = f(sampled_imgs) 
        X = np.stack(sampled_masks).astype(np.float32)
        y = probs[:,1]
        reg = 1e-3
        w, *_ = np.linalg.lstsq(X.T.dot(X) + reg*np.eye(num_segs), X.T.dot(y), rcond=None)
        heat = np.zeros_like(segments, dtype=np.float32)
        for seg_id in range(num_segs):
            heat[segments==seg_id] = w[seg_id]
        heat = (heat - heat.min()) / (heat.max()-heat.min()+1e-9)
        return heat
    
    def vit_attention_rollout(self, pil_image, discard_ratio=0.9):
        """Compute attention rollout for ViT-like models. Returns heatmap HxW"""
        try:
            x = self._preprocess_pil(pil_image).to(self.device)
            attn_weights = []
            def save_attn(module, inp, out):
                attn_weights.append(out.detach().cpu().numpy())
            hooks = []
            for name, module in self.model.named_modules():
                if name.endswith('attn') or 'attn' in name:
                    hooks.append(module.register_forward_hook(save_attn))
            _ = self.model(x)
            for h in hooks:
                h.remove()
            if len(attn_weights) == 0:
                raise RuntimeError('No attention weights captured')
            rollout = np.eye(attn_weights[0].shape[-1])
            for a in attn_weights:
                a_mean = a.mean(axis=1)[0]
                a_mean[a_mean < np.percentile(a_mean, discard_ratio*100)] = 0
                a_mean = a_mean + np.eye(a_mean.shape[0])
                a_mean = a_mean / a_mean.sum(axis=-1, keepdims=True)
                rollout = a_mean.dot(rollout)
            mask = rollout[0,1:]
            size = int(np.sqrt(mask.shape[0]))
            heat = mask.reshape(size, size)
            heat = cv2.resize(heat, (224,224))
            heat = (heat - heat.min()) / (heat.max()-heat.min()+1e-9)
            return heat
        except Exception as e:
            print('Attention rollout failed:', e)
            return np.zeros((224,224), dtype=np.float32)
        
def overlay_heatmap(pil_img, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.6):
    arr = np.array(pil_img.resize((heatmap.shape[1], heatmap.shape[0])))
    h = (heatmap*255).astype(np.uint8)
    cmap = cv2.applyColorMap(h, colormap)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha*cmap + (1-alpha)*arr).astype(np.uint8)
    return Image.fromarray(overlay)