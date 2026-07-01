import torch


def compute_saliency(model, x, target_class=None):
    """
    Compute saliency map for 1D spectral input
    Args:
        model: trained model
        x: tensor (1, C, L)
        target_class: optional class index

    Returns:
        saliency: numpy array (L,)
    """
    model.eval()

    x = x.clone().detach().requires_grad_(True)

    features = model.forward_features(x)
    logits = model.forward_logits(features)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[:, target_class]

    model.zero_grad(set_to_none=True)
    score.backward()

    saliency = x.grad.abs()

    saliency = saliency.max(dim=1)[0]

    saliency = saliency.squeeze()

    saliency = saliency / (saliency.max() + 1e-8)

    return saliency.cpu().numpy()


def compute_smoothgrad(model, x, target_class=None, n_samples=25, noise_std=0.02):
    """
    SmoothGrad for 1D spectral data

    Args:
        model: trained model
        x: tensor (1, C, L)
        target_class: optional class index
        n_samples: number of noisy samples
        noise_std: standard deviation of noise

    Returns:
        smoothed saliency (L,)
    """

    model.eval()

    x = x.clone().detach()

    smooth_saliency = 0

    # Determine target class ONCE before noise sampling
    if target_class is None:
        with torch.no_grad():
            features = model.forward_features(x)
            logits = model.forward_logits(features)
            target_class = logits.argmax(dim=1).item()
    for _ in range(n_samples):

        noise = torch.normal(mean=0, std=noise_std, size=x.shape).to(x.device)
        x_noisy = (x + noise).clone().detach().requires_grad_(True)

        features = model.forward_features(x_noisy)

        logits = model.forward_logits(features)

        score = logits[:, target_class]

        model.zero_grad(set_to_none=True)
        score.backward()

        grad = x_noisy.grad.abs()
        grad = grad.max(dim=1)[0]  # collapse channels

        smooth_saliency += grad

    smooth_saliency /= n_samples

    smooth_saliency = smooth_saliency.squeeze()

    smooth_saliency = smooth_saliency / (smooth_saliency.max() + 1e-8)

    return smooth_saliency.cpu().numpy()
