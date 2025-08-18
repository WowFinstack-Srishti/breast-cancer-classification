import cv2
import numpy as np

def normalize_staining(img, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization.
    img: RGB histopathology image (numpy array)
    Io: transmitted light intensity
    alpha: percentile cutoff for angle calculation
    beta: OD threshold for tissue detection
    """

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) + 1.0

    # Optical Density (OD)
    OD = -np.log(img / Io)
    OD = OD.reshape((-1, 3))  # Flatten into N x 3

    # Remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # SVD to find the 2 principal directions
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    eigvecs = eigvecs[:, [1, 0]]  # sort by largest eigenvalue

    # Project onto eigenvectors
    That = np.dot(ODhat, eigvecs)
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    HE = np.array([v1, v2]).T  # shape (3,2)

    # Concentrations
    C = np.dot(OD, np.linalg.pinv(HE))
    maxC = np.array([
        np.percentile(C[:, 0], 99),
        np.percentile(C[:, 1], 99)
    ], dtype=np.float32)
    C /= maxC

    # Reference H&E matrix
    HE_ref = np.array([[0.65, 0.70, 0.29],
                       [0.07, 0.99, 0.11]])  # shape (2,3)

    C2 = np.dot(C, HE_ref)

    # Recreate normalized image
    img_norm = Io * np.exp(-C2)
    img_norm[img_norm > 255] = 255
    img_norm = img_norm.reshape(img.shape).astype(np.uint8)

    return img_norm
