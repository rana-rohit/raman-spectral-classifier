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

    logits = model.forward_logits(x)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[:, target_class]

    model.zero_grad()
    score.backward()

    saliency = x.grad.abs()
    
    saliency = saliency.max(dim=1)[0]

    return saliency.squeeze().cpu().numpy()