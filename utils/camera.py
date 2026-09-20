"""
Camera opening helper.

OpenCV's default capture backend on Windows (MSMF) fails to read frames on
many machines — VideoCapture(0) reports opened=True but every read() returns
False. DirectShow usually works where MSMF doesn't, so try backends in order
and return the first that actually delivers a frame, rather than trusting
isOpened() alone.
"""

import cv2


def open_camera(camera_id=0, width=None, height=None, warmup_reads=5):
    """
    Open a camera, trying multiple backends until one actually returns frames.

    Args:
        camera_id: Camera device index
        width, height: Optional capture resolution to request
        warmup_reads: How many read() attempts to allow before declaring a
            backend dead (some cameras need a frame or two to start)

    Returns:
        cv2.VideoCapture that has been verified to return a frame

    Raises:
        RuntimeError: if no backend could read a frame
    """
    backends = [
        ('DSHOW', getattr(cv2, 'CAP_DSHOW', None)),
        ('MSMF', getattr(cv2, 'CAP_MSMF', None)),
        ('ANY', cv2.CAP_ANY),
    ]

    tried = []
    for name, backend in backends:
        if backend is None:
            continue

        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            tried.append(f"{name} (could not open)")
            cap.release()
            continue

        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(warmup_reads):
            ret, _frame = cap.read()
            if ret:
                print(f"[OK] Camera {camera_id} opened via {name}")
                return cap

        tried.append(f"{name} (opened but no frames)")
        cap.release()

    raise RuntimeError(
        f"Could not read frames from camera {camera_id}. Tried: {', '.join(tried)}. "
        "Check that no other application is using the camera and that camera "
        "access is enabled in Windows privacy settings."
    )
