"""Analytic checks for the CAD surface distance and cropping operations."""

import numpy as np

from hum_exo.geometry import capsule_surface_gaps, surface_crossings, surface_distance


def test_parallel_surfaces_and_remote_faces():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    first = vertices, faces
    shifted = vertices + np.array([0, 0, 0.005])
    second = np.r_[shifted, shifted + 100], np.r_[faces, faces + 3]
    assert surface_crossings(first, second) == 0
    assert np.isclose(surface_distance(first, second), 0.005)
    assert np.isclose(surface_distance(second, first), 0.005)
    unused = np.r_[shifted, vertices]
    # The retained source vertices on the other plane are not part of this cuff section.
    assert np.isclose(surface_distance(first, (unused, faces)), 0.005)


def test_crossing_triangles_and_interior_edge_minimum():
    faces = np.array([[0, 1, 2]])
    first = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -0.1, 0.0]]), faces
    second = np.array([[0.0, -2.0, 0.005], [0.0, 2.0, 0.005], [0.1, 0.0, 0.005]]), faces
    assert np.isclose(surface_distance(first, second), 0.005)
    crossing = np.array([[0.0, -0.05, -1.0], [0.0, -0.05, 1.0], [0.2, -0.05, 0.0]]), faces
    assert surface_crossings(first, crossing) > 0


def test_capsule_radius_contact_and_face_crossing():
    surface = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]), np.array([[0, 1, 2]])
    segments = np.array(
        [
            [[-0.1, 0.0, 0.006], [0.1, 0.0, 0.006]],
            [[-0.1, 0.0, 0.005], [0.1, 0.0, 0.005]],
            [[-0.1, 0.0, 0.004], [0.1, 0.0, 0.004]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]],
        ]
    )
    radii = np.full(5, 0.005)
    gaps = capsule_surface_gaps(surface, segments, radii, 0.01)
    np.testing.assert_allclose(gaps, [0.001, 0.0, -0.001, -0.005, 0.01], atol=1e-12)
