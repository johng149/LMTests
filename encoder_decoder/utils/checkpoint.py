import datetime
import os
import torch

def save_checkpoint(checkpoint_dir, checkpoint_name, model, losses, optm, tensorboard_log_dir=None):
    # model must have a `kwargs` attribute so that it can be re-instantiated
    assert hasattr(model, "kwargs")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    torch.save({
        'epoch': len(losses),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
        'losses': losses,
        'model_kwargs': model.kwargs,
        'tensorboard_log_dir': tensorboard_log_dir,
    }, checkpoint_path)

def load_checkpoint(checkpoint_dir, checkpoint_name, model_class, optm_class, device, map_location):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    optm = optm_class(model.parameters())
    optm.load_state_dict(checkpoint["optimizer_state_dict"])
    losses = checkpoint["losses"]
    log_dir = checkpoint["tensorboard_log_dir"]
    return model, optm, losses, log_dir

def try_loading(checkpoint_dir, checkpoint_name, model_class, optm_class, device, created_model_fallback_fn=None):
    try:
        model, optm, losses, log_dir = load_checkpoint(checkpoint_dir, checkpoint_name, model_class, optm_class, device)
        print(f"Resuming, have seen {len(losses)} epochs")
        print(f"Have {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
        print(f"Logging to {log_dir}")
        return model, optm, losses, log_dir
    except RuntimeError:
        # the checkpoint file probably exists but is on a different device,
        # can still probably load it, just need to specify the map_location
        print("Model found but saved on gpu, trying to load on cpu")
        model, optm, losses, log_dir = load_checkpoint(checkpoint_dir, checkpoint_name, model_class, optm_class, device, map_location="cpu")
        return model, optm, losses, log_dir
    except FileNotFoundError:
        if created_model_fallback_fn is None:
            raise FileNotFoundError(f"Could not find checkpoint file {checkpoint_name} in {checkpoint_dir}")
        else:
            print(f"Could not find checkpoint file {checkpoint_name} in {checkpoint_dir}, creating new model")
            model, optm = created_model_fallback_fn()
            print(f"Have {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
            losses = []
            now = datetime.datetime.now()
            log_dir = f"runs/run_at_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            print(f"Logging to {log_dir}")
            return model, optm, losses, log_dir