import cv2
import numpy as np

def normalize_staining(img, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization.
    img: RGB histopathology image (numpy array)
    Io: transmitted light intensity
    alpha: remove transparency
    beta: OD threshold for tissue detection
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) + 1.0

    OD = -np.log(img / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    eigvecs = eigvecs[:, [1, 0]]

    That = np.dot(ODhat, eigvecs)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    HE = np.array([v1, v2]).T

    C = np.dot(OD, np.linalg.pinv(HE))
    maxC = np.array([np.percentile(C[:, 0], 99), np.percentile(C[:, 1], 99)], dtype=np.float32)
    C = C / maxC

    HE_ref = np.array([[0.65, 0.70, 0.29],
                       [0.07, 0.99, 0.11]], dtype=np.float32)
    C2 = np.dot(C, HE_ref)

    img_norm = Io * np.exp(-C2)
    img_norm[img_norm > 255] = 255
    return img_norm.astype(np.uint8)