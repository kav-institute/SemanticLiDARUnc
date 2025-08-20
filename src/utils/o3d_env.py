import os, getpass

def ensure_o3d_runtime():
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

def has_display():
    # If there’s no X/Wayland display, Open3D can’t show a window.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))