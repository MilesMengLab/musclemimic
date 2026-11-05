"""Triangle-surface checks adapted from MyoAssist mounting tools.

Use rendered bones for installation clearance; collision capsules and the convex
hulls of hollow cuffs do not represent bone surfaces or cuff bores.
"""

import mujoco
import numpy as np
from scipy.spatial import cKDTree


def _crop_faces(surface: tuple, other_vertices: np.ndarray, padding: float) -> tuple:
    vertices, faces = surface
    triangles = vertices[faces]
    low, high = other_vertices.min(0) - padding, other_vertices.max(0) + padding
    keep = np.all(triangles.max(1) >= low, axis=1) & np.all(triangles.min(1) <= high, axis=1)
    faces = faces[keep]
    if not len(faces):
        return vertices[:0], faces
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def fitting_distance(first: tuple, second: tuple) -> float:
    """Fast vertex-to-nearby-face distance [m] used to bracket hinge fits.

    Final fits use full edge/edge and crossing checks. This approximation only
    chooses the neighborhood for the scalar solve on high-resolution CAD.
    """
    points, _ = first
    vertices, faces = second
    tree = cKDTree(vertices)
    adjacent = [[] for _ in vertices]
    for index, face in enumerate(faces):
        for vertex in face:
            adjacent[vertex].append(index)
    _, neighbors = tree.query(points, k=min(8, len(vertices)))
    best = np.inf
    triangles = vertices[faces]
    for point, ids in zip(points, np.asarray(neighbors).reshape(len(points), -1), strict=True):
        selected = np.unique([i for vertex in np.atleast_1d(ids) for i in adjacent[vertex]])
        if len(selected):
            best = min(best, float(_point_triangle_sq(point, triangles[selected]).min()))
    return float(np.sqrt(best))


def mesh_geometry(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rendered mesh vertices [m] and triangle indices in world axes."""
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    start = int(model.mesh_faceadr[mesh])
    faces = model.mesh_face[start : start + model.mesh_facenum[mesh]]
    return vertices, faces


def edge_hits(vertices: np.ndarray, faces: np.ndarray, triangles: np.ndarray) -> int:
    """Count mesh edges crossing another triangle surface."""
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    radius = float(radii.max())
    tree = cKDTree(centers)
    edges = np.unique(
        np.sort(
            np.concatenate(
                [
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )
    lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    points = vertices[edges]
    edges = edges[np.all(points.max(axis=1) >= lower, axis=1) & np.all(points.min(axis=1) <= upper, axis=1)]
    hits = 0
    for a, b in edges:
        origin = vertices[a]
        direction = vertices[b] - origin
        candidates = tree.query_ball_point((origin + vertices[b]) / 2, radius + np.linalg.norm(direction) / 2)
        if not candidates:
            continue
        candidates = np.asarray(candidates)
        candidates = candidates[
            np.linalg.norm(centers[candidates] - (origin + vertices[b]) / 2, axis=1)
            <= radii[candidates] + np.linalg.norm(direction) / 2 + 1e-12
        ]
        if not len(candidates):
            continue
        tri = triangles[candidates]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        h = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, h)
        mask = np.abs(det) > 1e-12
        if not np.any(mask):
            continue
        e1, e2, h, inv = e1[mask], e2[mask], h[mask], 1 / det[mask]
        delta = origin - tri[mask, 0]
        u = inv * np.einsum("ij,ij->i", delta, h)
        q = np.cross(delta, e1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits += bool(np.any((u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)))
    return hits


def surface_crossings(first: tuple, second: tuple) -> int:
    """Count crossings in both directions for two triangle surfaces."""
    va, fa = first
    vb, fb = second
    if np.any(va.max(axis=0) < vb.min(axis=0)) or np.any(vb.max(axis=0) < va.min(axis=0)):
        return 0
    first = _crop_faces(first, vb, 0.0)
    second = _crop_faces(second, va, 0.0)
    va, fa = first
    vb, fb = second
    if not len(fa) or not len(fb):
        return 0
    return edge_hits(va, fa, vb[fb]) + edge_hits(vb, fb, va[fa])


def _point_segment_sq(point, start, end):
    edge = end - start
    norm = np.sum(edge * edge, axis=-1)
    numerator = np.sum((point - start) * edge, axis=-1)
    t = np.divide(numerator, norm, out=np.zeros_like(numerator), where=norm > 1e-24)
    delta = point - (start + np.clip(t, 0, 1)[..., None] * edge)
    return np.sum(delta * delta, axis=-1)


def _point_triangle_sq(point, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    norm = np.sum(normal * normal, axis=1)
    signed = np.sum((point - a) * normal, axis=1)
    factor = np.divide(signed, norm, out=np.zeros_like(norm), where=norm > 1e-24)
    projected = point - factor[:, None] * normal
    inside = norm > 1e-24
    for start, end in ((a, b), (b, c), (c, a)):
        inside &= np.sum(np.cross(end - start, projected - start) * normal, axis=1) >= -1e-20
    edge_distance = np.minimum.reduce(
        [_point_segment_sq(point, a, b), _point_segment_sq(point, b, c), _point_segment_sq(point, c, a)]
    )
    return np.where(inside, factor * signed, edge_distance)


def surface_distance(first: tuple, second: tuple) -> float:
    """Minimum triangle-surface distance for disjoint meshes, in meters.

    Call surface_crossings separately before interpreting this as clearance.
    Includes vertex/face and interior edge/edge minima, not just nearest vertices.
    """
    va, fa = first
    vb, fb = second
    # Submesh audits may retain unused source vertices. Only referenced
    # vertices provide a valid upper bound on the selected triangle surfaces.
    upper = float(cKDTree(vb[np.unique(fb)]).query(va[np.unique(fa)])[0].min())
    # A vertex pair gives an upper bound. Faces beyond the other surface's
    # padded AABB cannot improve it, which removes most remote CAD triangles.
    first = _crop_faces(first, vb, upper)
    second = _crop_faces(second, va, upper)
    va, fa = first
    vb, fb = second
    triangles = vb[fb]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    bounds_low, bounds_high = triangles.min(1), triangles.max(1)
    tree = cKDTree(centers)
    max_radius = float(radii.max())
    best_sq = upper * upper
    for tri in va[fa]:
        center = tri.mean(axis=0)
        radius = float(np.linalg.norm(tri - center, axis=1).max())
        ids = tree.query_ball_point(center, radius + max_radius + np.sqrt(best_sq))
        if not ids:
            continue
        ids = np.asarray(ids)
        gap = np.maximum(0, np.maximum(tri.min(0) - bounds_high[ids], bounds_low[ids] - tri.max(0)))
        ids = ids[np.sum(gap * gap, axis=1) <= best_sq + 1e-20]
        if not len(ids):
            continue
        candidates = triangles[ids]
        dist = []
        for i in range(3):
            dist.append(_point_triangle_sq(tri[i], candidates))
            dist.append(_point_triangle_sq(candidates[:, i], np.broadcast_to(tri, candidates.shape)))
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            u = b - a
            for j in range(3):
                c, d = candidates[:, j], candidates[:, (j + 1) % 3]
                v, w = d - c, a - c
                aa = np.dot(u, u)
                bb, cc = v @ u, np.sum(v * v, axis=1)
                dd, ee = w @ u, np.sum(v * w, axis=1)
                denominator = aa * cc - bb * bb
                s = np.divide(bb * ee - cc * dd, denominator, out=np.zeros_like(cc), where=denominator > 1e-24)
                t = np.divide(aa * ee - bb * dd, denominator, out=np.zeros_like(cc), where=denominator > 1e-24)
                delta = w + s[:, None] * u - t[:, None] * v
                valid = (denominator > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
                dist.append(np.where(valid, np.sum(delta * delta, axis=1), np.inf))
        best_sq = min(best_sq, float(np.min(dist)))
    return float(np.sqrt(max(0, best_sq)))


def capsule_surface_gaps(surface: tuple, segments: np.ndarray, radii: np.ndarray, max_gap_m: float) -> np.ndarray:
    """Return capsule-to-triangle gaps [m], capped above at `max_gap_m`.

    Negative gaps indicate a capsule intersecting the mesh surface. Include
    segment/face intersections and interior segment/edge closest points.
    """
    vertices, faces = surface
    triangles = vertices[faces]
    low, high = triangles.min(1), triangles.max(1)
    result = np.full(len(segments), max_gap_m, dtype=float)
    mesh_low, mesh_high = vertices.min(0), vertices.max(0)
    for index, ((start, end), radius) in enumerate(zip(segments, radii, strict=True)):
        box_gap = np.maximum(0, np.maximum(np.minimum(start, end) - mesh_high, mesh_low - np.maximum(start, end)))
        if np.linalg.norm(box_gap) > max_gap_m + radius:
            continue
        gap = np.maximum(0, np.maximum(np.minimum(start, end) - high, low - np.maximum(start, end)))
        candidates = triangles[np.sum(gap * gap, axis=1) <= (max_gap_m + radius) ** 2]
        if not len(candidates):
            continue
        minimum = np.minimum(_point_triangle_sq(start, candidates), _point_triangle_sq(end, candidates))
        direction = end - start
        length_sq = np.dot(direction, direction)
        for j in range(3):
            a, b = candidates[:, j], candidates[:, (j + 1) % 3]
            minimum = np.minimum(minimum, _point_segment_sq(a, start, end))
            edge, delta = b - a, start - a
            dot = edge @ direction
            edge_sq = np.sum(edge * edge, axis=1)
            along = delta @ direction
            other = np.sum(edge * delta, axis=1)
            denominator = length_sq * edge_sq - dot * dot
            s = np.divide(dot * other - edge_sq * along, denominator, out=np.zeros_like(dot), where=denominator > 1e-24)
            t = np.divide(
                length_sq * other - dot * along, denominator, out=np.zeros_like(dot), where=denominator > 1e-24
            )
            distance = delta + s[:, None] * direction - t[:, None] * edge
            valid = (denominator > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
            minimum = np.minimum(minimum, np.where(valid, np.sum(distance * distance, axis=1), np.inf))
        # Moller-Trumbore segment/triangle intersections.
        e1, e2 = candidates[:, 1] - candidates[:, 0], candidates[:, 2] - candidates[:, 0]
        h = np.cross(direction, e2)
        determinant = np.sum(e1 * h, axis=1)
        inverse = np.divide(1, determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-12)
        offset = start - candidates[:, 0]
        u = inverse * np.sum(offset * h, axis=1)
        q = np.cross(offset, e1)
        v = inverse * (q @ direction)
        t = inverse * np.sum(e2 * q, axis=1)
        intersects = (np.abs(determinant) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        minimum[intersects] = 0
        result[index] = min(max_gap_m, float(np.sqrt(max(0, minimum.min()))) - radius)
    return result


VISUAL_SCENE_CAPACITY = 10000


def rendered_tendon_capsules(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_roots: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible muscle/tendon capsule endpoints [m] and radii [m]."""
    scene = mujoco.MjvScene(model, maxgeom=VISUAL_SCENE_CAPACITY)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    segments, radii = [], []
    allowed_tendons = None
    if body_roots is not None:
        roots = {model.body(name).id for name in body_roots}
        bodies = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor:
                if ancestor in roots:
                    bodies.add(body)
                    break
                ancestor = int(model.body_parentid[ancestor])
        allowed_tendons = {
            tendon
            for tendon in range(model.ntendon)
            if any(
                model.wrap_type[i] == mujoco.mjtWrap.mjWRAP_SITE and model.site_bodyid[model.wrap_objid[i]] in bodies
                for i in range(model.tendon_adr[tendon], model.tendon_adr[tendon] + model.tendon_num[tendon])
            )
        }
    for geom in scene.geoms[: scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_TENDON or geom.type != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        if allowed_tendons is not None and geom.objid not in allowed_tendons:
            continue
        half_axis = geom.mat.reshape(3, 3)[:, 2] * geom.size[2]
        segments.append([geom.pos - half_axis, geom.pos + half_axis])
        radii.append(geom.size[0])
    return np.asarray(segments).reshape(-1, 2, 3), np.asarray(radii)
