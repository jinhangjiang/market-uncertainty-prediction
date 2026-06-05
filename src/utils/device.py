import torch


def get_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_accelerator() -> str:
    """Return PyTorch Lightning accelerator string for NeuralForecast trainer_kwargs."""
    device = get_device()
    if device == "cuda":
        return "gpu"
    elif device == "mps":
        return "mps"
    return "cpu"


def get_trainer_kwargs(max_steps: int = 500) -> dict:
    """Return trainer_kwargs dict for NeuralForecast models."""
    accelerator = get_accelerator()
    kwargs = {
        "max_epochs": -1,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "accelerator": accelerator,
    }
    if accelerator == "gpu":
        kwargs["devices"] = 1
    return kwargs
