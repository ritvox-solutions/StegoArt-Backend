import torch.nn.functional as F

from ..config import ALPHA_COVER_LOSS, BETA_SECRET_LOSS


def combined_loss(
    cover,
    stego,
    secret,
    recovered_secret,
    alpha: float = ALPHA_COVER_LOSS,
    beta: float = BETA_SECRET_LOSS,
    secret_loss_type: str = "mse",
):
    """L_total = alpha * MSE(cover, stego) + beta * L_secret(secret, recovered_secret).

    secret_loss_type:
      "mse" (default) - works uniformly for image secrets and the text bit-map,
                         since both live in [0, 1].
      "bce"           - binary cross-entropy, an alternative for the
                         near-binary text bit-map.

    Returns (total, loss_cover, loss_secret) so each term can be logged
    separately during training.
    """
    loss_cover = F.mse_loss(stego, cover)

    if secret_loss_type == "mse":
        loss_secret = F.mse_loss(recovered_secret, secret)
    elif secret_loss_type == "bce":
        loss_secret = F.binary_cross_entropy(recovered_secret.clamp(1e-6, 1 - 1e-6), secret)
    else:
        raise ValueError(f"Unknown secret_loss_type: {secret_loss_type!r}")

    total = alpha * loss_cover + beta * loss_secret
    return total, loss_cover, loss_secret
