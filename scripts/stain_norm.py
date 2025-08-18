import cv2
import numpy as np

def normalize_staining(img, Io=240, alpha=1, beta=0.15):
    """
    Perform Macenko stain normalization.
    img: RGB histopathology image (numpy array, BGR if read by OpenCV)
    Io: transmitted light intensity
    alpha: percentile cutoff for angle calculation
    beta: OD threshold for tissue detection
    """

    # Convert BGR -> RGB and avoid log(0) by adding small epsilon
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) + 1.0

    # Step 1: Optical Density (OD)
    OD = -np.log((img + 1e-6) / Io)
    OD = OD.reshape((-1, 3))

    # Step 2: Remove transparent pixels (background)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if ODhat.size == 0:
        return img.astype(np.uint8)  # return original if no tissue detected

    # Step 3: SVD to find 2 principal stain vectors
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    eigvecs = eigvecs[:, [1, 0]]  # reorder: largest eigenvalue first

    # Step 4: Project ODhat onto eigenvectors
    That = np.dot(ODhat, eigvecs)
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    HE = np.array([v1, v2]).T  # shape (3,2)

    # Step 5: Estimate concentrations
    C = np.dot(OD, np.linalg.pinv(HE.T))

    # Normalize stain concentrations
    maxC = np.percentile(C, 99, axis=0)
    C /= (maxC + 1e-6)

    # Step 6: Reference H&E stain matrix (literature standard)
    HE_ref = np.array([[0.65, 0.70, 0.29],
                       [0.07, 0.99, 0.11]])  # shape (2,3)
    HE_ref = HE_ref.T  # make it (3,2)

    # Step 7: Reconstruct normalized image
    C2 = np.dot(C, HE_ref.T)
    img_norm = Io * np.exp(-C2)
    img_norm = np.clip(img_norm, 0, 255)
    img_norm = img_norm.reshape(img.shape).astype(np.uint8)

    return img_norm
