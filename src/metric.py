import numpy as np
import torch
from scipy.linalg import sqrtm


# Function to extract features from Inception v3
def extract_features(images, model):
    # Ensure images are on the right device (CUDA or CPU)
    images = images.cuda() if torch.cuda.is_available() else images
    # Get the features (use the last pooling layer before classification)
    with torch.no_grad():
        features = model(images)
    return features


def calculate_fid(real_features, generated_features):
    # Calculate the mean and covariance of real and generated feature distributions
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    # Calculate covariance matrices
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # Compute the Fréchet distance between real and generated distributions
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    # Numerically stable version of sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

ef calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    # Prevent division by zero or log of zero on perfect matches
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.mean().item()


def _gaussian_window(window_size, sigma):
    """Generate a 1D Gaussian kernel."""
    gauss = torch.exp(-torch.tensor([(x - window_size // 2) ** 2 for x in range(window_size)]) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    """Generate a 2D Gaussian window tensor for SSIM calculation."""
    _1D_window = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11, val_range=1.0):
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / den
    return ssim_map.mean().item()

class LPIPMetric:
    """
    A persistent wrapper for Learned Perceptual Image Patch Similarity (LPIPS).
    Instantiate this class ONCE before starting your evaluation loops.
    Requires the 'lpips' package (`pip install lpips`).
    """
    def __init__(self, net='alex', device='cuda'):
        """
        Args:
            net (str): Backing architecture network. Options: 'alex', 'vgg', or 'squeeze'.
            device (str): Execution target hardware ('cuda' or 'cpu').
        """
        import lpips
        self.loss_fn = lpips.LPIPS(net=net).to(device).eval()
        
    def __call__(self, img1, img2):
        # LPIPS natively expects features mapped within the [-1, 1] range
        img1_scaled = (img1 * 2.0) - 1.0
        img2_scaled = (img2 * 2.0) - 1.0
        
        with torch.no_grad():
            dist = self.loss_fn(img1_scaled, img2_scaled)
        return dist.mean().item()
