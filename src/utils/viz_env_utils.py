import os
def has_display() -> bool:
    # if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    #     return False
    # try:
    #     cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
    #     cv2.destroyWindow("__probe__")
    #     return True
    # except cv2.error:
    #     return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

def ensure_o3d_runtime():
    import getpass
    """
    Open3D sometimes requires XDG_RUNTIME_DIR (especially on Wayland).
    Create a private runtime dir if it's not set.
    """
    path = os.environ.get("XDG_RUNTIME_DIR")
    if not path:
        path = f"/tmp/runtime-{getpass.getuser()}"
        os.environ["XDG_RUNTIME_DIR"] = path
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    # must be 0700 permissions
    os.chmod(path, 0o700)