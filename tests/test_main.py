"""Tests for discrete1.py utility functions.

This module tests the helper functions in discrete1.main that build
angular quadratures, energy grids, velocity conversions, time steps,
and spatial material maps.
"""

import numpy as np
import pytest

import discrete1
from discrete1.main import gamma_time_steps


@pytest.mark.smoke
class TestAngularX:
    """Tests for angular_x function."""

    def test_basic_quadrature(self):
        """Test basic Gauss-Legendre quadrature generation."""
        angles = 4
        angle_x, angle_w = discrete1.angular_x(angles)

        # Check output shapes
        assert len(angle_x) == angles
        assert len(angle_w) == angles

        # Check weights are normalized
        assert np.isclose(np.sum(angle_w), 1.0)

        # Check ordinates are in [-1, 1]
        assert np.all(angle_x >= -1.0)
        assert np.all(angle_x <= 1.0)

    def test_vacuum_boundary(self):
        """Test with vacuum boundaries [0, 0]."""
        angles = 8
        angle_x, angle_w = discrete1.angular_x(angles, bc_x=[0, 0])

        assert len(angle_x) == angles
        assert np.isclose(np.sum(angle_w), 1.0)

    def test_left_reflective_boundary(self):
        """Test with left reflective boundary [1, 0]."""
        angles = 6
        angle_x, angle_w = discrete1.angular_x(angles, bc_x=[1, 0])

        # Check ordering is ascending
        assert np.all(np.diff(angle_x) >= 0)
        assert np.isclose(np.sum(angle_w), 1.0)

    def test_right_reflective_boundary(self):
        """Test with right reflective boundary [0, 1]."""
        angles = 6
        angle_x, angle_w = discrete1.angular_x(angles, bc_x=[0, 1])

        # Check ordering is descending
        assert np.all(np.diff(angle_x) <= 0)
        assert np.isclose(np.sum(angle_w), 1.0)

    def test_symmetry(self):
        """Test that Gauss-Legendre quadrature is symmetric."""
        angles = 8
        angle_x, angle_w = discrete1.angular_x(angles)

        # Weights should be symmetric
        assert np.allclose(angle_w, angle_w[::-1])


@pytest.mark.smoke
class TestEnergyGrid:
    """Tests for energy_grid function."""

    def test_grid_87_basic(self):
        """Test 87-group energy grid."""
        grid = 87
        groups_fine = 87
        edges_g, edges_gidx_fine = discrete1.energy_grid(grid, groups_fine)

        # Check edge array length
        assert len(edges_g) == grid + 1

        # Check index array
        assert len(edges_gidx_fine) == groups_fine + 1
        assert edges_gidx_fine[0] == 0
        assert edges_gidx_fine[-1] == grid

    def test_grid_361_basic(self):
        """Test 361-group energy grid."""
        grid = 361
        groups_fine = 361
        edges_g, edges_gidx_fine = discrete1.energy_grid(grid, groups_fine)

        assert len(edges_g) == grid + 1
        assert len(edges_gidx_fine) == groups_fine + 1

    def test_grid_618_basic(self):
        """Test 618-group energy grid."""
        grid = 618
        groups_fine = 618
        edges_g, edges_gidx_fine = discrete1.energy_grid(grid, groups_fine)

        assert len(edges_g) == grid + 1
        assert len(edges_gidx_fine) == groups_fine + 1

    def test_coarse_grid(self):
        """Test with coarse group specification."""
        grid = 87
        groups_fine = 87
        groups_coarse = 20

        edges_g, edges_gidx_fine, edges_gidx_coarse = discrete1.energy_grid(
            grid, groups_fine, groups_coarse
        )

        assert len(edges_g) == grid + 1
        assert len(edges_gidx_fine) == groups_fine + 1
        assert len(edges_gidx_coarse) == groups_coarse + 1
        assert edges_gidx_coarse[0] == 0
        assert edges_gidx_coarse[-1] == groups_fine

    def test_fine_group_subset(self):
        """Test with fewer fine groups than grid."""
        grid = 87
        groups_fine = 20

        edges_g, edges_gidx_fine = discrete1.energy_grid(grid, groups_fine)

        assert len(edges_g) == grid + 1
        assert len(edges_gidx_fine) == groups_fine + 1

    def test_optimize_flag(self):
        """Test optimize flag behavior."""
        grid = 87
        groups_fine = 87

        # With optimization
        edges_g1, edges_gidx_fine1 = discrete1.energy_grid(
            grid, groups_fine, optimize=True
        )

        # Without optimization
        edges_g2, edges_gidx_fine2 = discrete1.energy_grid(
            grid, groups_fine, optimize=False
        )

        # Results should be the same
        assert np.allclose(edges_g1, edges_g2)
        assert np.array_equal(edges_gidx_fine1, edges_gidx_fine2)

    def test_index_dtype(self):
        """Test that indices are returned as int32."""
        grid = 87
        groups_fine = 87
        edges_g, edges_gidx_fine = discrete1.energy_grid(grid, groups_fine)

        assert edges_gidx_fine.dtype == np.int32


@pytest.mark.smoke
class TestEnergyVelocity:
    """Tests for energy_velocity function."""

    def test_no_edges_provided(self):
        """Test with no energy edges provided."""
        groups = 10
        velocity = discrete1.energy_velocity(groups, edges_g=None)

        # Should return ones
        assert len(velocity) == groups
        assert np.allclose(velocity, 1.0)

    def test_with_energy_edges(self):
        """Test velocity calculation with energy edges."""
        groups = 5
        # Create sample energy edges (MeV)
        edges_g = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

        velocity = discrete1.energy_velocity(groups, edges_g=edges_g)

        # Check shape
        assert len(velocity) == groups

        # Velocities should be positive
        assert np.all(velocity > 0)

        # Higher energy should give higher velocity
        assert np.all(np.diff(velocity) > 0)

    def test_realistic_energy_grid(self):
        """Test with a realistic energy grid."""
        # Use actual grid from energy_grid function
        grid = 87
        groups_fine = 87
        edges_g, _ = discrete1.energy_grid(grid, groups_fine)

        velocity = discrete1.energy_velocity(groups_fine, edges_g=edges_g)

        assert len(velocity) == groups_fine
        assert np.all(velocity > 0)

    def test_single_group(self):
        """Test with single energy group."""
        groups = 1
        edges_g = np.array([1.0, 2.0])

        velocity = discrete1.energy_velocity(groups, edges_g=edges_g)

        assert len(velocity) == 1
        assert velocity[0] > 0


@pytest.mark.smoke
class TestGammaTimeSteps:
    """Tests for gamma_time_steps function."""

    def test_basic_half_steps(self):
        """Test basic half-step insertion."""
        edges_t = np.array([0.0, 1.0, 2.0, 3.0])
        combined = gamma_time_steps(edges_t, gamma=0.5, half_step=True)

        # Should have doubled length minus 1
        expected_length = len(edges_t) * 2 - 1
        assert len(combined) == expected_length

        # Should be sorted
        assert np.all(np.diff(combined) >= 0)

        # Original edges should be present
        for edge in edges_t:
            assert edge in combined

    def test_no_half_step_correction(self):
        """Test without half-step correction."""
        edges_t = np.array([0.0, 1.0, 2.0, 3.0])
        combined = gamma_time_steps(edges_t, gamma=0.5, half_step=False)

        expected_length = len(edges_t) * 2 - 1
        assert len(combined) == expected_length

    def test_different_gamma(self):
        """Test with different gamma value."""
        edges_t = np.array([0.0, 1.0, 2.0])
        combined = gamma_time_steps(edges_t, gamma=0.3, half_step=False)

        assert len(combined) == 5
        # Check that gamma step is correctly positioned
        assert np.isclose(combined[1], 0.3)
        assert np.isclose(combined[3], 1.3)

    def test_uniform_spacing(self):
        """Test with uniformly spaced time edges."""
        edges_t = np.linspace(0, 10, 11)
        combined = gamma_time_steps(edges_t, gamma=0.5, half_step=True)

        assert len(combined) == 21
        assert np.isclose(combined[0], 0.0)
        assert np.isclose(combined[-1], 10.0)

    def test_non_uniform_spacing(self):
        """Test with non-uniformly spaced time edges."""
        edges_t = np.array([0.0, 0.1, 0.5, 2.0, 5.0])
        combined = gamma_time_steps(edges_t, gamma=0.5, half_step=True)

        assert len(combined) == 9
        assert combined[0] == 0.0
        assert combined[-1] == 5.0


@pytest.mark.smoke
class TestSpatial1D:
    """Tests for spatial1d function."""

    def test_single_material(self):
        """Test with single material layer."""
        layers = [[0, "material_a", "0-10"]]
        edges_x = np.linspace(0, 10, 11)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == len(edges_x) - 1
        assert np.all(medium_map == 0)
        assert medium_map.dtype == np.int32

    def test_two_materials(self):
        """Test with two material layers."""
        layers = [[0, "material_a", "0-5"], [1, "material_b", "5-10"]]
        edges_x = np.linspace(0, 10, 11)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == 10
        assert np.all(medium_map[:5] == 0)
        assert np.all(medium_map[5:] == 1)

    def test_multiple_regions(self):
        """Test material with multiple regions."""
        layers = [[0, "material_a", "0-2, 4-6"], [1, "material_b", "2-4, 6-8"]]
        edges_x = np.linspace(0, 8, 9)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == 8
        assert np.all(medium_map[:2] == 0)
        assert np.all(medium_map[2:4] == 1)
        assert np.all(medium_map[4:6] == 0)
        assert np.all(medium_map[6:8] == 1)

    def test_labels_mode(self):
        """Test with labels=True (returns float array)."""
        layers = [[0, "material_a", "0-10"]]
        edges_x = np.linspace(0, 10, 11)

        medium_map = discrete1.spatial1d(layers, edges_x, labels=True)

        assert len(medium_map) == 10
        # Should not be int32 when labels=True
        assert medium_map.dtype != np.int32

    def test_check_assertion(self):
        """Test that check=True catches incomplete fills."""
        layers = [[0, "material_a", "0-5"]]  # Only fills half
        edges_x = np.linspace(0, 10, 11)

        # Should raise assertion error for unfilled cells
        with pytest.raises(AssertionError):
            discrete1.spatial1d(layers, edges_x, check=True)

    def test_no_check(self):
        """Test with check=False allows incomplete fills."""
        layers = [[0, "material_a", "0-5"]]
        edges_x = np.linspace(0, 10, 11)

        medium_map = discrete1.spatial1d(layers, edges_x, check=False)

        # Should have -1 in unfilled cells
        assert np.any(medium_map == -1)
        assert np.all(medium_map[:5] == 0)
        assert np.all(medium_map[5:] == -1)

    def test_complex_layering(self):
        """Test with complex multi-layer structure."""
        layers = [[0, "fuel", "0-1, 3-4, 6-7"], [1, "moderator", "1-3, 4-6, 7-10"]]
        edges_x = np.linspace(0, 10, 101)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == 100
        # Check no unfilled cells
        assert np.all(medium_map != -1)

    def test_whitespace_handling(self):
        """Test that whitespace in region strings is handled."""
        layers = [[0, "material_a", " 0 - 5 "], [1, "material_b", " 5 - 10 "]]
        edges_x = np.linspace(0, 10, 11)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == 10
        assert np.all(medium_map[:5] == 0)
        assert np.all(medium_map[5:] == 1)

    def test_fine_mesh(self):
        """Test with fine spatial mesh."""
        layers = [[0, "material_a", "0-10"]]
        edges_x = np.linspace(0, 10, 1001)

        medium_map = discrete1.spatial1d(layers, edges_x)

        assert len(medium_map) == 1000
        assert np.all(medium_map == 0)


@pytest.mark.smoke
class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_problem_setup(self):
        """Test setting up a complete 1D problem."""
        # Angular
        angles = 8
        angle_x, angle_w = discrete1.angular_x(angles, bc_x=[0, 0])

        # Energy
        grid = 87
        groups = 20
        edges_g, edges_gidx = discrete1.energy_grid(grid, groups)

        # Velocity
        velocity = discrete1.energy_velocity(groups, edges_g=None)

        # Time
        edges_t = np.linspace(0, 1, 11)
        combined_t = gamma_time_steps(edges_t)

        # Spatial
        layers = [[0, "fuel", "0-5"], [1, "reflector", "5-10"]]
        edges_x = np.linspace(0, 10, 51)
        medium_map = discrete1.spatial1d(layers, edges_x)

        # Verify all components
        assert len(angle_x) == angles
        assert len(edges_g) == grid + 1
        assert len(velocity) == groups
        assert len(combined_t) == 21
        assert len(medium_map) == 50

    def test_energy_velocity_with_real_grid(self):
        """Test velocity calculation with actual energy grid data."""
        grid = 87
        groups = 87
        edges_g, _ = discrete1.energy_grid(grid, groups)
        velocity = discrete1.energy_velocity(groups, edges_g)

        # Verify reasonable velocity values for neutrons
        # Speed should be less than speed of light
        assert np.all(velocity < 3e10)  # cm/s
        # Speed should be positive
        assert np.all(velocity > 0)
