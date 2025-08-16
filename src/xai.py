"""
Polished XAI utilities:
- Grad-CAM via Captum (LayerGradCam)
- Guided Grad-CAM (combination) for visualization
- GradientExplainer/DeepExplainer wrapper for Deep models
- Superpixel-based KernelSHAP for images (uses skimage.slic)
- Attention rollout for transformers (ViT)
Notes:
- KernelSHAP on pixels is prohibitively slow; superpixel approach reduces complexity.
- GradientExplainer (shap) or DeepExplainer is preferred for deep models when available.
"""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop
import shap
from skimage.segmentation import slic

# basic transform
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
        """
        model: the PyTorch model object. If your model stores backbone as `.net` (ResNet wrapper used in project),
               the code will attempt to use model.net for LayerGradCam. For ViT-type models set model_type='vit'
               and provide a target_layer if needed (or use attention_rollout).
        device: torch.device
        model_type: 'resnet' or 'vit'
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.model_type = model_type
        if target_layer is None and model_type == 'resnet':
            # default last conv for ResNet
            try:
                self.target_layer = self.model.net.layer4[-1].conv3
            except Exception:
                self.target_layer = None
        else:
            self.target_layer = target_layer

    def _preprocess_pil(self, pil):
        return transform(pil).unsqueeze(0)

    def gradcam(self, pil_image, target_class=None, upsample_size=(224,224)):
        """Return normalized heatmap (H,W) in [0,1]"""
        inp = self._preprocess_pil(pil_image).to(self.device)
        net_for_attr = self.model.net if hasattr(self.model,'net') else self.model
        if self.target_layer is None:
            raise ValueError('target_layer must be provided for gradcam with this model')
        lgc = LayerGradCam(net_for_attr, self.target_layer)
        with torch.no_grad():
            logits = net_for_attr(inp)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
        attr = lgc.attribute(inp, target=target_class)
        # interpolate to desired size
        attr = LayerAttribution.interpolate(attr, upsample_size)
        heat = attr.squeeze().cpu().detach().numpy()
        heat = np.maximum(heat, 0.0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        return heat

    def guided_gradcam(self, pil_image, target_class=None):
        """Return guided gradcam map (H,W,C normalized 0..1)"""
        # guided backprop
        inp = self._preprocess_pil(pil_image).to(self.device)
        net_for_attr = self.model.net if hasattr(self.model,'net') else self.model
        gbp = GuidedBackprop(net_for_attr)
        with torch.no_grad():
            logits = net_for_attr(inp)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())
        gb = gbp.attribute(inp, target=target_class)
        gb = gb.squeeze().cpu().detach().numpy()
        gb = np.transpose(gb, (1,2,0))
        gb = (gb - gb.min()) / (gb.max()-gb.min()+1e-9)
        # gradcam
        gc = self.gradcam(pil_image, target_class)
        gc = np.expand_dims(gc, axis=2)
        guided = gb * gc
        guided = (guided - guided.min()) / (guided.max()-guided.min()+1e-9)
        return guided

    def shap_gradient_explainer(self, pil_image, background_images, nsamples=50):
        """
        Try shap.GradientExplainer (fast). If it fails, fallback to superpixel KernelSHAP.
        background_images: list of PIL images (small set, e.g., 5-20)
        Returns heatmap HxW normalized.
        """
        try:
            # try GradientExplainer (works with some torch models)
            def f(x):
                # x: batch HWC numpy
                x_t = torch.tensor(x).permute(0,3,1,2).float().to(self.device)
                with torch.no_grad():
                    out = self.model(x_t)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs

            bg_np = np.stack([to_numpy_img(im) for im in background_images]).astype(np.float32)
            explainer = shap.GradientExplainer(f, bg_np[:min(len(bg_np),10)])
            inp_np = to_numpy_img(pil_image)[None].astype(np.float32)
            shap_vals = explainer.shap_values(inp_np, nsamples=nsamples)
            # shap_vals: classes x H x W x C
            sv = np.sum(np.array(shap_vals), axis=-1)
            sv = sv[0]
            sv = (sv - sv.min()) / (sv.max()-sv.min()+1e-9)
            return sv
        except Exception as e:
            # fallback to KernelSHAP on superpixels
            print('GradientExplainer failed, falling back to superpixel KernelSHAP:', e)
            return self.shap_superpixel(pil_image, n_segments=100, nsamples=200)

    def shap_superpixel(self, pil_image, n_segments=100, nsamples=200):
        """
        Superpixel KernelSHAP: segment image into superpixels and estimate importance per segment.
        Returns upsampled per-pixel heatmap.
        NOTE: This is an approximation and uses linear regression to estimate segment weights.
        """
        img_np = to_numpy_img(pil_image)
        segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
        num_segs = segments.max() + 1
        avg_color = img_np.mean(axis=(0,1))
        # background for KernelSHAP: mean image of dataset or black
        def masker(mask_vec):
            # segments_mask is a binary vector of kept segments (1 means keep)
            out = img_np.copy()
            for seg_id in range(num_segs):
                if mask_vec[seg_id] == 0:
                    out[segments == seg_id] = avg_color
            return out

        # build masker samples
        # random sample of masks
        import random
        X = []
        y = []
        B = min(50, nsamples) 
        masks_sampled = []
        for _ in range(nsamples):
            mask = np.random.choice([0,1], size=(num_segs,), p=[0.5,0.5]).astype(np.float32)
            masks_sampled.append(mask)
        imgs_to_eval = [masker(m) for m in masks_sampled]

        # model function for shap (takes list of images HWC numpy)
        # get model outputs, # nsamples x num_classes
        # approximate per-segment importance via linear regression
        # Build X matrix (nsamples x num_segs)
        # target: prob for positive class (assume class 1)
        # solve least squares with regularization
        # map segment weights to pixels
        def f(images):
            x = np.stack(images).astype(np.float32)
            x_tensor = torch.tensor(x).permute(0,3,1,2).float().to(self.device)
            with torch.no_grad():
                out = self.model(x_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            return probs[:,1]  
        probs = f(imgs_to_eval)
        X = np.stack(masks_sampled).astype(np.float32)
        y = probs.astype(np.float32)
     
        lam = 1e-3
        A = X.T.dot(X) + lam * np.eye(num_segs)
        b = X.T.dot(y)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
      
        heat = np.zeros_like(segments, dtype=np.float32)
        for seg_id in range(num_segs):
            heat[segments == seg_id] = w[seg_id]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        return heat

    def vit_attention_rollout(self, pil_image, discard_ratio=0.9):
        """
        Attention rollout for ViT-like models. Works if model exposes attention maps during forward.
        This implementation tries to attach hooks to layers named with 'attn' or similar (best-effort).
        Returns heatmap 224x224 in [0,1].
        """
        try:
            x = self._preprocess_pil(pil_image).to(self.device)
            # get attentions via forward hook if model provides
            attn_weights = []
            def save_attn(module, inp, out):
                # out expected shape: (B, num_heads, seq_len, seq_len)
                if isinstance(out, tuple):
                    out = out[0]
                attn_weights.append(out.detach().cpu().numpy())
            # Attempt to attach to transformer blocks (timm ViT uses blocks)
            hooks = []
            for name, module in self.model.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    try:
                        hooks.append(module.register_forward_hook(save_attn))
                    except Exception:
                        pass
            _ = self.model(x)
            for h in hooks:
                h.remove()
            if len(attn_weights) == 0:
                raise RuntimeError('No attention weights captured.')
            # attn_weights: list of arrays
            # rollout: start with identity
            rollout = np.eye(attn_weights[0].shape[-1])
            for a in attn_weights:
                # average heads
                a_mean = a.mean(axis=1)[0]  
                a_mean = a_mean + np.eye(a_mean.shape[0])
                a_mean = a_mean / a_mean.sum(axis=-1, keepdims=True)
                rollout = a_mean.dot(rollout)
            # take rollout for class token attention to patches
            mask = rollout[0,1:] 
            # reshape to square
            size = int(np.sqrt(mask.shape[0]))
            heat = mask.reshape(size, size)
            heat = cv2.resize(heat, (224,224))
            heat = (heat - heat.min()) / (heat.max()-heat.min()+1e-9)
            return heat
        except Exception as e:
            print('Attention rollout failed:', e)
            return np.zeros((224,224), dtype=np.float32)

# utility: overlay heatmap on pil image
def overlay_heatmap(pil_img, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.6):
    arr = np.array(pil_img.resize((heatmap.shape[1], heatmap.shape[0])))
    h = (heatmap*255).astype(np.uint8)
    cmap = cv2.applyColorMap(h, colormap)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha*cmap + (1-alpha)*arr).astype(np.uint8)
    return Image.fromarray(overlay)