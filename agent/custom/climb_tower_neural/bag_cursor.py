"""Give Win32's per-gesture cursor restoration a neutral bag position."""
import ctypes
from ctypes import wintypes
from maa.define import MaaWin32InputMethodEnum
from utils import logger as logger_module


logger = logger_module.get_logger(__name__)


def park_bag_cursor(context):
    info = context.tasker.controller.info
    if info["type"] != "win32" or info.get("mouse_method") not in (
            MaaWin32InputMethodEnum.SendMessageWithCursorPos,
            MaaWin32InputMethodEnum.PostMessageWithCursorPos):
        return
    try:
        _park_window_cursor(info["hwnd"])
    except OSError as error:
        logger.debug("Could not park the bag cursor: %s", error)


def _park_window_cursor(hwnd):
    # SendMessageWithCursorPos saves the physical cursor at touch_down and
    # restores it at touch_up. Set its resting point before the bag gestures,
    # rather than adding a TouchMove (which carries MK_LBUTTON in Maa 5.16.1).
    user32 = ctypes.WinDLL("user32")
    user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    user32.SetCursorPos.restype = wintypes.BOOL
    user32.GetPhysicalCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetPhysicalCursorPos.restype = wintypes.BOOL
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    previous = user32.SetThreadDpiAwarenessContext(-4)
    try:
        cursor = wintypes.POINT()
        if not user32.GetPhysicalCursorPos(ctypes.byref(cursor)):
            return
        hovered = user32.WindowFromPoint(cursor)
        if user32.GetAncestor(hovered, 2) != user32.GetAncestor(hwnd, 2):
            return
        point = wintypes.POINT(20, 20)
        if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
            return
        user32.SetCursorPos(point.x, point.y)
    finally:
        user32.SetThreadDpiAwarenessContext(previous)
