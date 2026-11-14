"""Surface meshes from masks, in RAS millimetres, with no smoothing, and OBJ export."""

from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes


def mask_mesh(
    mask: np.ndarray, affine: np.ndarray, step_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Surface of a mask. Vertices are in RAS mm, so meshes line up with the CT's position."""
    found = np.argwhere(mask)
    if found.size == 0:
        raise ValueError("Cannot build a mesh from an empty mask")
    low = found.min(axis=0)
    high = found.max(axis=0) + 1
    window = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
    padded = np.pad(mask[window].astype(np.float32), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step_size)
    voxels = vertices - 1 + low
    world = (affine @ np.c_[voxels, np.ones(len(voxels))].T).T[:, :3]
    return world, faces.astype(np.int64)


def mesh_volume_mm3(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of a closed mesh, used to check a mesh against its mask."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, name: str) -> None:
    """Write an OBJ file. An existing file is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(f"# StoneKit research software. Not for clinical use. Units: RAS millimetres.\no {name}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b, c in faces + 1:
            f.write(f"f {a} {b} {c}\n")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read back an OBJ written by write_obj."""
    vertices, faces = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                faces.append([int(v) - 1 for v in parts[1:4]])
    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)
