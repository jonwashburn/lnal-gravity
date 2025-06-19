#!/usr/bin/env python3
"""Test the fixed solver"""

import numpy as np
from rs_gravity_tunable_enhanced import EnhancedGravitySolver, GalaxyData, GalaxyParameters

# Test parameters from optimization
params = {
    'lambda_eff': 5.6485598737362877e-05,
    'beta_scale': 1.2851759613930136,
    'mu_scale': 0.5594128078850475,
    'coupling_scale': 1.125621989144501
}

# Create solver
solver = EnhancedGravitySolver(**params)

# Test on a simple galaxy
R_kpc = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
v_obs = np.array([50, 80, 100, 110, 115, 118])
v_err = np.array([2, 3, 3, 4, 4, 5])
sigma_gas = np.array([20, 15, 10, 7, 5, 3])
sigma_disk = np.array([100, 80, 60, 40, 30, 20])

galaxy = GalaxyData(
    name="Test",
    R_kpc=R_kpc,
    v_obs=v_obs,
    v_err=v_err,
    sigma_gas=sigma_gas,
    sigma_disk=sigma_disk
)

print("Testing solver...")
try:
    result = solver.solve_galaxy(galaxy)
    print(f"Success! χ²/N = {result['chi2_reduced']:.2f}")
    print(f"v_model = {result['v_model']}")
except Exception as e:
    print(f"Error: {e}")

# Test kernel functions directly
print("\nTesting kernels...")
r_test = np.array([0.1, 1.0, 10.0, 100.0]) * 3.086e19  # kpc to m
for r in r_test:
    try:
        G_r = solver.G_running(r)
        F_r = solver.F_kernel(r)
        print(f"r = {r/3.086e19:.1f} kpc: G/G_inf = {G_r/6.674e-11:.3e}, F = {F_r:.3f}")
    except Exception as e:
        print(f"Error at r = {r/3.086e19:.1f} kpc: {e}") 