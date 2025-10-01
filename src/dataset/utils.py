import numpy as np
import cv2

def rotate_z(point_cloud, angle):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply the rotation matrix to the point cloud
    rotated_cloud = np.dot(point_cloud, rotation_matrix)
    
    return rotated_cloud


def rotate_equirectangular_image(image, angle):
    # Calculate the shift amount based on the angle of rotation
    shift_amount = int(round((angle / (2 * np.pi)) * image.shape[1]))

    # Roll the image around the width axis
    rotated_image = np.roll(image, shift_amount, axis=1)

    return rotated_image

def build_normal_xyz(xyz, norm_factor=0.25, ksize = 3):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
    '''
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]

    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0/norm_factor)    
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0/norm_factor)

    #build cross product
    normal = -np.dstack((Syx*Szy - Szx*Syy,
                        Szx*Sxy - Szy*Sxx,
                        Sxx*Syy - Syx*Sxy))

    # normalize corss product
    n = np.linalg.norm(normal, axis=2)+1e-10
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    return normal

def to_deflection_coordinates(x,y,z):
    # To cylindrical
    p = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # To spherical   
    theta = -np.arctan2(p, z) + np.pi/2
    return phi, theta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

def _fit_texts_inside_axes(ax, texts, log_scale: bool, margin_px: int = 10, max_iters: int = 5):
    """Expand ylim so all annotation texts are fully inside the axes area."""
    fig = ax.figure
    for _ in range(max_iters):
        fig.canvas.draw()  # ensure bboxes are current
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()

        # highest text top (in pixels)
        max_overflow_px = 0.0
        max_text_top_data = None

        for t in texts:
            bb = t.get_window_extent(renderer=renderer)
            overflow_px = bb.y1 - (ax_bb.y1 - margin_px)
            if overflow_px > max_overflow_px:
                max_overflow_px = overflow_px
                # convert bbox top (x doesn't matter) to data y
                x_pix = (bb.x0 + bb.x1) * 0.5
                _, y_data = inv.transform((x_pix, bb.y1))
                max_text_top_data = y_data

        if max_overflow_px <= 0:
            break  # everything fits

        # expand ylim
        y0, y1 = ax.get_ylim()
        if max_text_top_data is None:
            # fallback: conservative bump
            y1 = y1 * (1.15 if log_scale else 1.05)
        else:
            # set top so that the tallest text sits below the top by margin
            # add a small safety factor
            safety = 1.05 if not log_scale else 1.10
            y1 = max(y1, max_text_top_data) * safety

        ax.set_ylim(y0, y1)

def plot_pointCounts_per_class(
    class_counts: dict,
    class_names: list,
    num_classes: int,
    color_map: dict,
    ignore_ids=(0,),
    log_scale=True,
    sort_by_count=False,
    figsize=(20, 7),
    title="Dataset Class Distribution",
    title_fontsize=22,
    label_fontsize=14,
    tick_fontsize=12,
    annot_fontsize=12,
    annot_rotation=0,
    annot_offset=12,
    top_pad=None,              # if None: auto-fit (recommended). If number: force factor.
    add_white_outline=True,
    save_path=None,             # save figure if given
    dpi=200                     # dpi for saving
):
    ids = [i for i in range(num_classes) if i not in set(ignore_ids)]
    counts = [int(class_counts.get(i, 0)) for i in ids]
    names  = [class_names[i] for i in ids]
    colors = [np.array(color_map[i]) / 255.0 for i in ids]

    df = pd.DataFrame({"class_id": ids, "class": names, "count": counts})
    if sort_by_count:
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        colors = [np.array(color_map[i]) / 255.0 for i in df["class_id"].tolist()]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(data=df, x="class", y="count", palette=colors)

    # labels
    ax.set_title(title, fontsize=title_fontsize, pad=24, weight="bold")
    ax.set_xlabel("Class", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Number of points", fontsize=label_fontsize, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Number of points (log scale)", fontsize=label_fontsize, labelpad=10)

    # initial ylim guess
    ymax = max([c for c in counts if c > 0] or [1])
    ymin = 1 if log_scale else 0
    ax.set_ylim(ymin, ymax * (1.15 if top_pad is None else top_pad))

    # annotations (clip_on=True so we see if they overflow)
    texts = []
    for p in ax.patches:
        h = p.get_height()
        if h <= 0:
            continue
        txt = ax.annotate(
            f"{int(h):,}",
            (p.get_x() + p.get_width()/2., h),
            xytext=(0, annot_offset), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=annot_fontsize, fontweight="bold",
            rotation=annot_rotation,
            clip_on=True  # keep text inside axes; we'll expand ylim below
        )
        if add_white_outline:
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
        texts.append(txt)

    # AUTO-FIT so no text crosses the top grid
    if top_pad is None:
        _fit_texts_inside_axes(ax, texts, log_scale=log_scale, margin_px=20, max_iters=20)
    else:
        # respect fixed top_pad
        ax.set_ylim(ymin, ymax * top_pad)

    # layout (do after fitting)
    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
    

# def spherical_projection(pc,
#                          height=64,
#                          width=2048,
#                          fov_up_deg=2.0,
#                          fov_down_deg=-24.8,
#                          range_thresh=1.0,
#                          max_range=None):
#     """
#     Project NxC point cloud to an equirectangular image (H, W, C).

#     - Horizontal (azimuth φ): full 360°, periodic, pixel-centered bins.
#     - Vertical (elevation θ): fixed FoV [fov_down, fov_up] in degrees.

#     Args:
#         pc:  (N, C) array, first three columns are x,y,z (meters).
#         height: rows (elevation bins).
#         width:  columns (azimuth bins).
#         fov_up_deg:   +2.0 for HDL-64 (KITTI).
#         fov_down_deg: -24.8 for HDL-64 (KITTI).
#         range_thresh: drop points with r <= this (meters).
#         max_range:    optional max range clip (meters).

#     Returns:
#         img:   (H, W, C) float32
#         alpha: (H, W) dummy grid (kept for compatibility)
#         (theta_min, theta_max), (phi_min, phi_max)
#     """
#     x = pc[:, 0]; y = pc[:, 1]; z = pc[:, 2]
#     r = np.sqrt(x*x + y*y + z*z)

#     keep = r > range_thresh
#     if max_range is not None:
#         keep &= (r <= max_range)
#     if not np.any(keep):
#         img = np.zeros((height, width, pc.shape[1]), dtype=np.float32)
#         alpha = np.zeros((height, width), dtype=np.float32)
#         return img, alpha, (np.radians(fov_down_deg), np.radians(fov_up_deg)), (-np.pi, np.pi)

#     x = x[keep]; y = y[keep]; z = z[keep]; r = r[keep]
#     pts = pc[keep]

#     # ---- angles ----
#     phi   = np.arctan2(y, x)                 # [-pi, pi]
#     elev  = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))  # [-pi/2, pi/2]

#     # ---- vertical mapping with fixed FoV ----
#     fov_up   = np.radians(fov_up_deg)
#     fov_down = np.radians(fov_down_deg)
#     fov = fov_up - fov_down  # positive

#     # v in [0, H-1], top row = fov_up
#     v = (1.0 - (elev - fov_down) / fov) * height
#     v = np.floor(v).astype(np.int32)
#     v = np.clip(v, 0, height - 1)

#     # ---- horizontal mapping (periodic), pixel-centered, no mirroring ----
#     # u in [0, W-1]; use (phi + pi) / (2*pi) * W, then floor & modulo
#     u = ((phi + np.pi) / (2.0 * np.pi)) * width
#     u = np.floor(u).astype(np.int32) % width  # periodic wrap

#     # ---- depth ordering: write far->near so near overwrites far ----
#     order = np.argsort(r)          # near first
#     order = order[::-1]            # far ... near
#     u = u[order]; v = v[order]; pts = pts[order]

#     img = np.zeros((height, width, pc.shape[1]), dtype=np.float32)
#     img[v, u, :] = pts

#     # keep compatibility fields (not used downstream)
#     theta_min, theta_max = fov_down, fov_up
#     phi_min, phi_max = -np.pi, np.pi
#     # simple angle grid (optional)
#     theta_lin = np.linspace(theta_min, theta_max, height, endpoint=False) + 0.5*(theta_max-theta_min)/height
#     phi_lin   = np.linspace(phi_min, phi_max,   width,  endpoint=False) + 0.5*(phi_max-phi_min)/width
#     theta_img = np.repeat(theta_lin[:, None], width, axis=1)
#     phi_img   = np.repeat(phi_lin[None, :], height, axis=0)
#     alpha = np.sqrt(theta_img**2 + phi_img**2).astype(np.float32)

#     return img, alpha, (theta_min, theta_max), (phi_min, phi_max)



# OLD spehrical projection
# With descending bins the index mapping is wrong, so many points get assigned to mirrored columns
# -> you get a duplicate panorama (one real, one mirrored/shifted). When you later roll/flip, the artifact becomes very obvious.

def spherical_projection(pc, height=64, width=2048, theta_range=None, th=1.0, sort_largest_first=False, bins_h=None, max_range=None):
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*C
    Returns:
        pj_img: projected spherical iamges, shape: h*w*C
    '''

    # filter all small range values to avoid overflows in theta min max calculation
    #if isinstance(theta_range, type(None)):
        
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    arr1inds = r.argsort()
    if sort_largest_first:
        pc = pc[arr1inds]
    else:
        pc = pc[arr1inds[::-1]]
    #pc = pc[arr1inds]
    # r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    # if not isinstance(max_range,type(None)):
    #     indices = np.where((r > th)*(r<=max_range))
    # else:
    #     indices = np.where(r > th)
    # pc = pc[indices]
        
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
    phi, theta = to_deflection_coordinates(x,y,z)

    #indices = np.where(r > th)
    if isinstance(theta_range, type(None)):
        theta_min, theta_max = [theta.min(), theta.max()]
    else: 
        theta_min, theta_max = theta_range
        
    phi_min, phi_max = [-np.pi, np.pi]
    
    # assuming uniform distribution of rays
    if isinstance(bins_h, type(None)):
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
        
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]
    
    theta_img = np.stack(width*[bins_h], axis=-1)
    phi_img = np.stack(height*[bins_w], axis=0)

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1
    
    pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)

    
    pj_img[idx_h, idx_w, :] = pc

   
    alpha = np.sqrt(np.square(theta_img)+np.square(phi_img))
   
    return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max) 
