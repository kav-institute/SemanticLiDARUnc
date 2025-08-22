# models/inputs.py
import torch

def set_model_inputs(range_img, reflectivity, xyz, normals, cfg):
    """
    Build a list of inputs for the selected baseline.

    Returns:
        [main]               for single-input nets (e.g., SalsaNext)
        [main, metadata]     for two-input nets (e.g., Reichert)
    """
    baseline = cfg["model_settings"]["baseline"].lower()

    main_channels = [range_img]
    if cfg["model_settings"].get("reflectivity", 0):
        main_channels.append(reflectivity)

    if baseline in {"salsanext", "salsanextadf"}:
        main_channels.append(xyz)
        if cfg["model_settings"].get("normals", 0):
            main_channels.append(normals)
        main = torch.cat(main_channels, dim=1)
        return [main]

    elif baseline == "reichert":
        main = torch.cat(main_channels, dim=1)
        if cfg["model_settings"].get("normals", 0):
            metadata = torch.cat([xyz, normals], dim=1)
        else:
            metadata = xyz
        return [main, metadata]

    else:
        raise ValueError(f"Unknown baseline: {cfg['model_settings']['baseline']}")
