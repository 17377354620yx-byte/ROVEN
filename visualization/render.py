"""Headless Open3D renderer with a shared camera for comparable panels."""

import os
import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

# Must be set before importing Open3D to enable rendering on headless servers.
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


DEFAULT_CAMERA = {
    "direction": np.asarray([1.0, -1.35, 0.85], dtype=np.float64),
    "up": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "fov_deg": 36.0,
}


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have shape (N, 3) and be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def make_camera(
    point_sets: Sequence[np.ndarray],
    *,
    direction: Optional[Sequence[float]] = None,
    up: Optional[Sequence[float]] = None,
    fov_deg: float = 36.0,
) -> Dict[str, np.ndarray]:
    """Build one deterministic camera enclosing all supplied point sets."""
    points = np.concatenate([_points(p, "point set") for p in point_sets], axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) * 0.5
    diagonal = float(np.linalg.norm(upper - lower))
    if diagonal <= np.finfo(np.float64).eps:
        diagonal = 1.0
    view = np.asarray(direction if direction is not None else DEFAULT_CAMERA["direction"],
                      dtype=np.float64)
    view /= np.linalg.norm(view)
    camera_up = np.asarray(up if up is not None else DEFAULT_CAMERA["up"],
                           dtype=np.float64)
    camera_up /= np.linalg.norm(camera_up)
    # The factor leaves enough white margin for wide liver geometries.
    eye = center + view * diagonal * 1.4
    return {"center": center, "eye": eye, "up": camera_up,
            "fov_deg": float(fov_deg)}


def _render_open3d(
    geometries: Sequence[Tuple[np.ndarray, np.ndarray]],
    camera: Mapping[str, np.ndarray],
    *,
    width: int = 760,
    height: int = 620,
    point_size: float = 3.0,
    background: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
    renderer.scene.set_background(np.asarray(background, dtype=np.float32))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = float(point_size)

    for index, (raw_points, raw_colors) in enumerate(geometries):
        points = _points(raw_points, f"geometry[{index}] points")
        colors = np.asarray(raw_colors, dtype=np.float64)
        if colors.shape == (3,):
            colors = np.broadcast_to(colors, points.shape).copy()
        if colors.shape != points.shape:
            raise ValueError(f"geometry[{index}] colors must have shape (N, 3) or (3,)")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        renderer.scene.add_geometry(f"cloud_{index}", cloud, material)

    renderer.scene.camera.look_at(
        np.asarray(camera["center"], dtype=np.float32),
        np.asarray(camera["eye"], dtype=np.float32),
        np.asarray(camera["up"], dtype=np.float32),
    )
    image = np.asarray(renderer.render_to_image())
    # Explicit destruction avoids accumulating GPU renderer contexts in batches.
    del renderer
    return image[:, :, :3].astype(np.uint8, copy=False)


def _render_software(
    geometries: Sequence[Tuple[np.ndarray, np.ndarray]],
    camera: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
    point_size: float,
    background: Sequence[float],
) -> np.ndarray:
    """Deterministic perspective fallback for Open3D/driver incompatibilities."""
    eye = np.asarray(camera["eye"], dtype=np.float64)
    center = np.asarray(camera["center"], dtype=np.float64)
    nominal_up = np.asarray(camera["up"], dtype=np.float64)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, nominal_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = np.tan(np.deg2rad(float(camera["fov_deg"])) * 0.5)
    aspect = float(width) / float(height)

    projected = []
    for raw_points, raw_colors in geometries:
        points = _points(raw_points, "geometry points")
        colors = np.asarray(raw_colors, dtype=np.float64)
        if colors.shape == (3,):
            colors = np.broadcast_to(colors, points.shape)
        relative = points - eye
        depth = relative @ forward
        valid = depth > 1e-8
        x = (relative @ right) / np.maximum(depth, 1e-8) / (tangent * aspect)
        y = (relative @ up) / np.maximum(depth, 1e-8) / tangent
        valid &= (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)
        px = (x[valid] * 0.5 + 0.5) * (width - 1)
        py = (0.5 - y[valid] * 0.5) * (height - 1)
        projected.extend(zip(depth[valid], px, py, colors[valid]))

    rgb = tuple(int(round(255 * float(v))) for v in background[:3])
    canvas = Image.new("RGB", (int(width), int(height)), rgb)
    draw = ImageDraw.Draw(canvas)
    radius = max(1.0, float(point_size) * 0.5)
    # Painter's algorithm: nearby samples overwrite distant samples.
    for _, x, y, color in sorted(projected, key=lambda item: item[0], reverse=True):
        fill = tuple(int(round(255 * value)) for value in np.clip(color, 0.0, 1.0))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return np.asarray(canvas)


def resolve_render_backend(backend: str = "auto") -> str:
    """Resolve ``auto`` without entering a known process-fatal renderer."""
    if backend not in ("auto", "open3d", "software"):
        raise ValueError("backend must be 'auto', 'open3d', or 'software'")
    selected = backend
    if selected == "auto":
        o3d_version = tuple(int(part) for part in o3d.__version__.split(".")[:2])
        numpy_major = int(np.__version__.split(".")[0])
        selected = "software" if (o3d_version <= (0, 16) and numpy_major >= 2) else "open3d"
        if selected == "software":
            warnings.warn(
                "Open3D 0.16 with NumPy 2.x has a process-fatal rendering ABI issue; "
                "using the deterministic fallback renderer. Upgrade Open3D or use "
                "NumPy < 2 to enable the Open3D backend.", RuntimeWarning, stacklevel=2
            )
    return selected


def render_point_clouds(
    geometries: Sequence[Tuple[np.ndarray, np.ndarray]],
    camera: Mapping[str, np.ndarray],
    *,
    width: int = 760,
    height: int = 620,
    point_size: float = 3.0,
    background: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    backend: str = "auto",
) -> np.ndarray:
    """Render colored clouds, preferring Open3D and safely falling back.

    Open3D 0.16's renderer is process-fatal with NumPy 2.x. ``auto`` detects
    that known ABI combination before entering Filament. Set ``backend`` to
    ``open3d`` to require Open3D or ``software`` for deterministic CI output.
    """
    selected = resolve_render_backend(backend)
    kwargs = dict(width=width, height=height, point_size=point_size,
                  background=background)
    if selected == "open3d":
        return _render_open3d(geometries, camera, **kwargs)
    return _render_software(geometries, camera, **kwargs)
