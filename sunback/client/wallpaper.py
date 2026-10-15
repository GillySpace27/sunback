"""Platform wallpaper setters (SB-21), moved out of sunback/putter/DesktopPutter.py.

DesktopPutter.update_background calls set_wallpaper, so the old client keeps the
exact same behaviour. Only the macOS path has ever been verified on a real
machine (the 0.6.17.3 NSWorkspace fix); the Windows and Linux paths are moved
unchanged and have not been run on either platform since.
"""
import logging
import os
import platform
import subprocess

logger = logging.getLogger(__name__)


def screens():
    """The macOS screens (NSScreen objects), or [] where AppKit is unavailable."""
    try:
        from AppKit import NSScreen
    except ImportError:
        return []
    return list(NSScreen.screens() or [])


def set_wallpaper_macos(path, screen=None):
    """Set the macOS desktop picture on one screen, or on every screen when screen is None.

    Uses the native NSWorkspace API (via pyobjc) instead of driving System
    Events with osascript. On macOS 14+ the osascript path needs Automation
    permission, which a background LaunchAgent (lingon) can't be prompted
    to grant, so it silently no-ops. NSWorkspace runs in-process in the
    user's GUI session and returns an explicit success/error, so we trust
    that instead of reading the wallpaper back (the macOS getter lags by a
    set and can't be used to verify). Falls back to osascript only if
    pyobjc isn't installed (and then only for every screen at once).
    """
    path = str(path)
    try:
        from AppKit import NSWorkspace, NSScreen
        from Foundation import NSURL
    except ImportError:
        # ponytail: last-resort fallback; NSWorkspace is the real path.
        cmd = f'tell application "System Events" to tell every desktop to set picture to "{path}"'
        subprocess.run(["osascript", "-e", cmd], check=True)
        return

    url = NSURL.fileURLWithPath_(path)
    ws = NSWorkspace.sharedWorkspace()
    targets = [screen] if screen is not None else NSScreen.screens()
    if not targets:
        raise OSError("No screens available: is there a GUI login session?")
    for scr in targets:
        ok, err = ws.setDesktopImageURL_forScreen_options_error_(url, scr, {}, None)
        if not ok:
            raise OSError(f"NSWorkspace failed to set wallpaper: {err}")


def set_wallpaper_windows(path):
    """Windows: the registry value plus UpdatePerUserSystemParameters. Not tested on Windows."""
    try:
        # Update registry and refresh the wallpaper
        command = [
            "REG",
            "ADD",
            r"HKCU\Control Panel\Desktop",
            "/V",
            "Wallpaper",
            "/T",
            "REG_SZ",
            "/F",
            "/D",
            str(path),
        ]
        logger.info("Executing command: %s", " ".join(command))
        subprocess.run(command, check=True)

        command_refresh = ["RUNDLL32.EXE", "user32.dll,UpdatePerUserSystemParameters"]
        logger.info("Executing command: %s", " ".join(command_refresh))
        subprocess.run(command_refresh, check=True)

        logger.info("Wallpaper updated successfully.")

    except subprocess.CalledProcessError as e:
        logger.error("Failed to update wallpaper on Windows: %s", e)
        raise OSError(f"Failed to update wallpaper on Windows: {e}")


def set_wallpaper_linux(path):
    """Linux: gsettings on GNOME; KDE and XFCE are not implemented. Not tested on Linux."""
    try:
        desktop_env = os.getenv("XDG_CURRENT_DESKTOP", "").lower()
        if "gnome" in desktop_env:
            subprocess.run(
                ["gsettings", "set", "org.gnome.desktop.background", "picture-uri", f"file://{path}"],
                check=True,
            )
        elif "kde" in desktop_env:
            raise NotImplementedError("KDE wallpaper update is not implemented.")
        elif "xfce" in desktop_env:
            raise NotImplementedError("XFCE wallpaper update is not implemented.")
        else:
            raise OSError(f"Unsupported desktop environment: {desktop_env}")
    except subprocess.CalledProcessError as e:
        raise OSError(f"Failed to update wallpaper on Linux: {e}")


def set_wallpaper(path, screen=None):
    """Dispatch on the operating system; screen applies to macOS only."""
    this_system = platform.system()
    if this_system == "Darwin":
        set_wallpaper_macos(path, screen)
    elif this_system == "Windows":
        set_wallpaper_windows(path)
    elif this_system == "Linux":
        set_wallpaper_linux(path)
    else:
        raise OSError(f"Unsupported operating system: {this_system}")
