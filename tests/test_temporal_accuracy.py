"""Temporal accuracy tests using manufactured solutions.

These tests verify time integration accuracy for schemes like BE and BDF2
using manufactured solutions as references.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import timed1d
from discrete1.utils import manufactured as mms
from tests import problems


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.bdf1
def test_backward_euler_01():
    # Spatial
    cells_x = 200
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = discrete1.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.0

    for steps in [50, 75, 100]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        parameters = problems.manufactured_td_01(cells_x, angles, edges_t, temporal=1)
        approx = timed1d.backward_euler(*parameters, steps, dt)

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = mms.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab
@pytest.mark.bdf1
def test_backward_euler_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = discrete1.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.0

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        parameters = problems.manufactured_td_02(cells_x, angles, edges_t, temporal=1)
        approx = timed1d.backward_euler(*parameters, steps, dt)

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = mms.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 1 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.bdf2
def test_bdf2_01():
    # Spatial
    cells_x = 200
    length_x = 2.0
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = discrete1.angular_x(angles)

    error_x = []
    error_y = []
    T = 200.0

    for steps in [50, 100, 200]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        parameters = problems.manufactured_td_01(cells_x, angles, edges_t, temporal=3)
        approx = timed1d.bdf2(*parameters, steps, dt)

        exact = mms.solution_td_01(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = mms.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)


@pytest.mark.slab
@pytest.mark.bdf2
def test_bdf2_02():
    # Spatial
    cells_x = 100
    length_x = np.pi
    edges_x = np.linspace(0, length_x, cells_x + 1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angles = 4
    angle_x, angle_w = discrete1.angular_x(angles)

    error_x = []
    error_y = []
    T = 20.0

    for steps in [40, 60, 80]:
        dt = T / (steps)
        edges_t = np.linspace(0, T, steps + 1)

        parameters = problems.manufactured_td_02(cells_x, angles, edges_t, temporal=3)
        approx = timed1d.bdf2(*parameters, steps, dt)

        exact = mms.solution_td_02(centers_x, angle_x, edges_t[1:])
        exact = np.sum(exact * angle_w[None, None, :, None], axis=2)

        err = np.linalg.norm(approx[-1] - exact[-1])

        error_y.append(err)
        error_x.append(dt)

    atol = 5e-2
    for ii in range(len(error_x) - 1):
        ratio = error_x[ii] / error_x[ii + 1]
        accuracy = mms.order_accuracy(error_y[ii], error_y[ii + 1], ratio)
        assert 2 - accuracy < atol, "Accuracy: " + str(accuracy)
