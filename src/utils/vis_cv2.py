# utils/vis_cv2.py
import numpy as np
import cv2

def add_horizontal_uncertainty_colorbar(image, num_classes, colormap=cv2.COLORMAP_TURBO, height=20,
                                        num_ticks=5, font_scale=0.7, thickness=1, color=(225, 225, 225)):
    max_uncertainty = np.log(num_classes)
    width = image.shape[1]
    gradient = np.linspace(0, max_uncertainty, width).astype(np.float32).reshape(1, -1)
    grad255 = np.clip((gradient / max_uncertainty) * 255.0, 0, 255).astype(np.uint8)
    bar = cv2.applyColorMap(cv2.resize(grad255, (width, height), interpolation=cv2.INTER_LINEAR), colormap)

    labels = ["Certain", "Confident", "Ambiguous", "Doubtful", "Uncertain"]
    for i in range(num_ticks):
        x = int(i * (width - 1) / (num_ticks - 1))
        label = labels[i]
        ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if i <= 2:
            text_x = x
        elif i == 2:
            text_x = x
        else:
            text_x = x - ts[0]
        cv2.putText(bar, label, (text_x, ts[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return np.concatenate((image, bar), axis=0)

def visualize_semantic_segmentation_cv2(mask, class_colors):
    """
    Visualize semantic segmentation mask using class colors with cv2.

    Parameters:
    - mask: 2D NumPy array containing class IDs for each pixel.
    - class_colors: Dictionary mapping class IDs to BGR colors.

    Returns:
    - visualization: Colored semantic segmentation image in BGR format.
    """
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        vis[mask == class_id] = color
    return vis

def open_window(name="inf", width=1024, height=700):
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        try:
            cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, 0)  # CV_WINDOW_FREERATIO
        except Exception:
            pass
        return True
    except cv2.error:
        return False

def close_window(name="inf"):
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass

def _fit_within(h, w, max_w, max_h, scale=1.0):
    """
    Scale (h,w) by 'scale' but clamp to (max_w, max_h) preserving aspect ratio.
    Returns (disp_w, disp_h, effective_scale).
    """
    req_w, req_h = int(round(w * scale)), int(round(h * scale))
    if req_w == 0 or req_h == 0:
        return 1, 1, 1.0
    cap = min(max_w / req_w, max_h / req_h, 1.0)  # never upscale beyond requested
    eff = scale * cap
    return int(round(w * eff)), int(round(h * eff)), eff

def show_stack(images, scale=1.5, ensure_even=True, name="inf", max_window=(1280, 800)):
    """
    images: iterable of equally-wide HxWx3 arrays
    scale : requested up/downscale for display
    max_window: (max_width, max_height) hard cap for the window/image
    """
    img = np.vstack(images)
    if ensure_even:
        h, w = img.shape[:2]
        if h % 2: img = img[:-1]
        if w % 2: img = img[:, :-1]

    h, w = img.shape[:2]
    disp_w, disp_h, eff = _fit_within(h, w, max_window[0], max_window[1], scale=scale)
    if eff != 1.0:
        img_disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_disp = img

    cv2.imshow(name, img_disp)
    cv2.resizeWindow(name, img_disp.shape[1], img_disp.shape[0])  # keep window equal to clamped image
    return img_disp  # (optionally inspect img_disp.shape or eff if needed)

# def show_stack(images, scale=1.5, ensure_even=True, name="inf"):
#     """
#     images: iterable of equally-wide HxWx3 arrays
#     scale : up/downscale factor for display
#     """
#     img = np.vstack(images)
#     if ensure_even:                      # some WMs dislike odd sizes
#         h, w = img.shape[:2]
#         if h % 2: img = img[:-1]
#         if w % 2: img = img[:, :-1]
#     if scale != 1.0:
#         h, w = img.shape[:2]
#         img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
#     cv2.imshow(name, img)
#     # optional: adjust window to image size
#     cv2.resizeWindow(name, img.shape[1], img.shape[0])
#     return img
