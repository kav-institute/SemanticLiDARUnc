# utils/viz_panel.py
import cv2, numpy as np
import os
from typing import Dict, List, Callable, Optional, Tuple, Union

def _stack_vertical(images: List[np.ndarray]) -> np.ndarray:
    """Stack HxWx3 uint8 images vertically, resizing widths to the minimum width."""
    if not images:
        return np.zeros((1, 2, 3), np.uint8)
    W = min(im.shape[1] for im in images)
    if any(im.shape[1] != W for im in images):
        out = []
        for im in images:
            h, w = im.shape[:2]
            nh = int(round(h * (W / max(1, w))))
            out.append(cv2.resize(im, (W, nh), interpolation=cv2.INTER_NEAREST))
        images = out
    return np.vstack(images)

class VizPanel:
    """
    Right-side checkbox UI with:
      - lazy optional layers (builders called only when ticked)
      - panel width auto-fits to label text
      - window height auto-fits to show ALL layers and ALL selected images
      - no blank padding
    """
    def __init__(self,
                window_name: str = "inf",
                panel_width: int = 180,               # initial; will be auto-fitted each frame
                max_window: Tuple[int,int] = (1920, 1080),   # used only for initial base fit
                font = cv2.FONT_HERSHEY_SIMPLEX,
                create_window: bool = True):
        self.window = window_name
        self.max_w, self.max_h = max_window
        self.font = font

        # auto-fit panel width each frame; keep sensible clamps
        self.panel_w_min = 140
        self.panel_w_max = 420
        self.panel_w     = int(panel_width)

        # checkbox row metrics
        self.row_h   = 28
        self.box_sz  = 18
        self.left_pad= 12
        self.top_pad = 10

        # persistent state
        self.enabled: Dict[str, bool] = {}
        self.order:   List[str]       = []
        self.hit_boxes: Dict[str, Tuple[int,int,int,int]] = {}

        # base scale for first fit; user zoom if you want it
        self._base_eff: Optional[float] = None
        self._user_scale: float = 1.0
        self._zoom_step: float  = 1.05

        # window lifetime
        self._window_created = False
        self._want_window    = bool(create_window)

        # mouse map for hit-tests after resize
        self._mouse_map = {"scale_x":1.0, "scale_y":1.0, "x_off0":0}
    
    # ---------- public helpers ----------
    def set_default_enabled(self, defaults: Dict[str, bool]):
        for n, v in defaults.items():
            if n not in self.enabled:
                self.enabled[n] = bool(v)

    def refit_next_frame(self):
        """Force recomputation of base fit (dataset or monitor changed)."""
        self._base_eff = None

    def destroy(self):
        if self._window_created:
            try:
                cv2.destroyWindow(self.window)
            except Exception:
                pass
            self._window_created = False

    def handle_key(self, key: int):
        # optional zoom hotkeys
        if key in (ord('+'), ord('=')): self._zoom(self._zoom_step)
        if key == ord('-'):             self._zoom(1.0/self._zoom_step)
        if key == ord('0'):             self._zoom_reset()

    # ---------- internals ----------
    # def _ensure_window(self):
    #     if self._window_created or not self._want_window:
    #         return
    #     cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
    #     cv2.setMouseCallback(self.window, self._on_mouse)
    #     self._window_created = True
    def _ensure_window(self):
        if self._window_created or not self._want_window:
            return
        # No resize handles, no toolbars; client area = image size
        flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(self.window, flags)
        # Make sure Qt doesn’t keep aspect/letterbox
        try:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
            cv2.setWindowProperty(self.window, cv2.WND_PROP_AUTOSIZE, 1)
        except Exception:
            pass
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._window_created = True

    def _zoom(self, factor: float, lo: float = 0.25, hi: float = 4.0):
        self._user_scale = float(np.clip(self._user_scale * factor, lo, hi))
        self._user_scale = round(self._user_scale * 100.0) / 100.0  # snap to 1%

    def _zoom_reset(self):
        self._user_scale = 1.0

    def _on_mouse(self, event, x, y, flags, userdata):
        if not self._window_created:
            return
        sx = self._mouse_map["scale_x"]; sy = self._mouse_map["scale_y"]
        x0 = int(round(x / max(1e-12, sx)))
        y0 = int(round(y / max(1e-12, sy)))

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # toggles (right panel only)
        if x0 < self._mouse_map["x_off0"]:
            return
        for name, (bx1,by1,bx2,by2) in self.hit_boxes.items():
            if bx1 <= x0 <= bx2 and by1 <= y0 <= by2:
                self.enabled[name] = not self.enabled.get(name, True)
                return

    def _auto_panel_width(self) -> int:
        """Fit width to longest wrapped label segment (split on underscores)."""
        max_px = 0
        for name in self.order:
            for part in name.split('_'):
                w,_ = cv2.getTextSize(part, self.font, 0.7, 1)[0]
                max_px = max(max_px, w)
        needed = self.left_pad + self.box_sz + 8 + max_px + 12
        return int(np.clip(needed, self.panel_w_min, self.panel_w_max))

    def _draw_wrapped_label(self, img, text, x, y, max_width):
        parts = text.split('_')
        line = ""
        y_off = 0
        for p in parts:
            trial = p if line == "" else (line + "_" + p)
            w,_ = cv2.getTextSize(trial, self.font, 0.7, 1)[0]
            if w > max_width and line:
                cv2.putText(img, line, (x, y + y_off), self.font, 0.7, (225,225,225), 1, cv2.LINE_AA)
                line = p
                y_off += 16
            else:
                line = trial
        if line:
            cv2.putText(img, line, (x, y + y_off), self.font, 0.7, (225,225,225), 1, cv2.LINE_AA)

    def _panel_needed_height(self, names: List[str]) -> int:
        header_h = 54
        # estimate wrapped rows (1–2 lines); we still step by row_h visually
        return self.top_pad + header_h + len(names) * self.row_h + 12

    def _panel_img(self, H: int, x_off0: int) -> np.ndarray:
        panel = np.full((H, self.panel_w, 3), 34, np.uint8)

        # Header
        cv2.putText(panel, "Layers", (self.left_pad, 26 + self.top_pad),
                    self.font, 1.2, (240,240,240), 2, cv2.LINE_AA)

        # Checkbox list (all of them; no footer, no cropping)
        self.hit_boxes.clear()
        y = self.top_pad + 54  # just below header

        for name in self.order:
            bx1 = self.left_pad
            by1 = y - self.box_sz + 5
            bx2 = bx1 + self.box_sz
            by2 = by1 + self.box_sz
            self.hit_boxes[name] = (x_off0 + bx1, by1, x_off0 + bx2, by2)

            cv2.rectangle(panel, (bx1, by1), (bx2, by2), (200,200,200), 1)
            if self.enabled.get(name, True):
                cv2.line(panel, (bx1+3, by1+9), (bx1+7, by1+13), (72,220,72), 2)
                cv2.line(panel, (bx1+7, by1+13), (bx1+14, by1+5), (72,220,72), 2)

            max_text_w = self.panel_w - (bx2 + 14)
            self._draw_wrapped_label(panel, name, bx2 + 8, y, max_text_w)
            y += self.row_h

        return panel

    # ---------- render ----------
    def render_with_builders(self,
                            base_sources: Dict[str, Union[np.ndarray, Callable[[], np.ndarray]]],
                            optional_builders: Dict[str, Callable[[], np.ndarray]],
                            scale: float = 1.5) -> np.ndarray:
        """Build selected images and render panel (auto-sized, no padding)."""
        self._ensure_window()

        # reconcile visible names
        current = list(base_sources.keys()) + list(optional_builders.keys())
        if self.order:
            self.order = [n for n in self.order if n in current]
        for n in list(self.enabled.keys()):
            if n not in current: del self.enabled[n]
        for n in current:
            if n not in self.order:   self.order.append(n)
            if n not in self.enabled: self.enabled[n] = True

        # build left images lazily (only enabled)
        left_imgs: List[np.ndarray] = []
        for n in self.order:
            if not self.enabled.get(n, True):
                continue
            if n in base_sources:
                v = base_sources[n]
                im = v() if callable(v) else v
                left_imgs.append(im)
            elif n in optional_builders:
                im = optional_builders[n]()  # built ONLY when ticked
                left_imgs.append(im)

        left_full = _stack_vertical(left_imgs)
        LH, LW = left_full.shape[:2]

        # auto-fit panel width and total height to include ALL labels
        self.panel_w = self._auto_panel_width()
        x_off0 = LW
        panel_h_needed = self._panel_needed_height(self.order)
        H = max(LH, panel_h_needed)   # show all content; no scroll/pad

        # if left is shorter than H, extend it with a tiny strip (not visible)
        if LH < H:
            pad = np.full((H - LH, LW, 3), 32, np.uint8)
            left_full = np.vstack([left_full, pad])

        panel = self._panel_img(H, x_off0)
        composed0 = np.hstack([left_full, panel])

        # base fit once (to keep initial window reasonable); afterwards only user zoom changes scale
        if self._base_eff is None:
            cw, ch = composed0.shape[1], composed0.shape[0]
            eff_w = self.max_w / max(1, cw)
            eff_h = self.max_h / max(1, ch)
            self._base_eff = min(1.0, eff_w, eff_h) * scale

        eff = self._base_eff * self._user_scale
        disp_w = int(round(composed0.shape[1] * eff))
        disp_h = int(round(composed0.shape[0] * eff))
        composed = cv2.resize(composed0, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # update mouse map
        self._mouse_map = {
            "scale_x": disp_w / max(1, composed0.shape[1]),
            "scale_y": disp_h / max(1, composed0.shape[0]),
            "x_off0": x_off0
        }

        if self._window_created:
            cv2.imshow(self.window, composed)
            cv2.resizeWindow(self.window, composed.shape[1], composed.shape[0])

        return composed

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
# Make global instance of class VizPanel #
_PANEL: Optional[VizPanel] = None       #<------ VizPanel object
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

try:
    from viz_env_utils import has_display, ensure_o3d_runtime
except:
    from utils.viz_env_utils import has_display, ensure_o3d_runtime
# ---------- Panel utils -------------
def _create_panel(create_window: bool):
    global _PANEL
    _PANEL = VizPanel(
        window_name="inf",
        panel_width=180,
        max_window=(1280, 800),
        create_window=create_window
    )

def get_panel() -> VizPanel:
    """Singleton panel; headless-safe (no window if no display)."""
    global _PANEL
    if _PANEL is None:
        _create_panel(create_window=has_display())
    return _PANEL  # type: ignore

def destroy_panel():
    global _PANEL
    if _PANEL is not None:
        _PANEL.destroy()
        _PANEL = None

def register_optional_names(names, default_enabled=False):
    """
    Make checkboxes appear even when they start disabled.
    Call once in Trainer.__init__.
    """
    p = get_panel()
    p.set_default_enabled({n: bool(default_enabled) for n in names})
    p.refit_next_frame()

def create_ia_plots(
    base_images_dict: Dict[str, np.ndarray],
    optional_builders: Dict[str, Callable[[], np.ndarray]],
    args_o3d: Tuple[np.ndarray, np.ndarray],
    save_dir: str = "",
    enable: bool = True,
    scale: float = 1.5,
):
    """
    NOTE: pass ALL optional names in optional_builders (don't filter by enabled);
    VizPanel will call the lambda only if the box is ticked.
    """
    if not enable:
        destroy_panel()
        return

    panel = get_panel()
    panel.render_with_builders(
        base_sources=base_images_dict,
        optional_builders=optional_builders,
        scale=scale
    )

    key = cv2.waitKey(1) & 0xFF
    if key != 0xFF:
        panel.handle_key(key)

    if key == ord("q"):
        try:
            import open3d as o3d
            if not has_display(): return
            ensure_o3d_runtime()
            xyz, color_bgr = args_o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
            rgb = (color_bgr[..., ::-1].reshape(-1, 3).astype(np.float32)) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([mesh, pcd])
        except Exception:
            pass