#!/usr/bin/env python3
"""
Test Recognition Science gravity equation against real galaxy rotation data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
import os
import glob

# Physical constants
c = 2.998e8  # m/s
G_inf = 6.674e-11  # m³/kg/s²
kpc_to_m = 3.086e19  # m/kpc
km_to_m = 1000  # m/km
Msun = 1.989e30  # kg
Lsun = 3.828e26  # W

# Recognition Science constants
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
beta = -(phi - 1) / phi**5  # ≈ -0.0557
lambda_eff = 60e-6  # m (60 μm)
g_dagger = 1.2e-10  # m/s² (MOND acceleration scale)

# Recognition lengths (from kernel poles)
ell_1 = 0.97  # kpc
ell_2 = 24.3  # kpc

# Information field parameters
I_star = 4.0e18  # J/m³
mu_param = 1.0  # Dimensionless

def read_galaxy_data(filename):
    """Read galaxy rotation data from SPARC format file"""
    data = np.loadtxt(filename, skiprows=3)
    
    # Extract columns
    r = data[:, 0]  # kpc
    v_obs = data[:, 1]  # km/s
    v_err = data[:, 2]  # km/s
    v_gas = data[:, 3]  # km/s
    v_disk = data[:, 4]  # km/s
    v_bulge = data[:, 5]  # km/s
    
    # Calculate total baryonic velocity
    v_bar_squared = v_gas**2 + v_disk**2 + v_bulge**2
    v_bar = np.sqrt(v_bar_squared)
    
    # Extract distance from header
    with open(filename, 'r') as f:
        first_line = f.readline()
        distance = float(first_line.split('=')[1].split('Mpc')[0].strip())
    
    return {
        'r': r,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_bar': v_bar,
        'distance': distance,
        'name': os.path.basename(filename).replace('_rotmod.dat', '')
    }

def Xi_function(u):
    """The Xi function from Recognition Science"""
    if np.any(u <= 0):
        return np.zeros_like(u)
    return (np.exp(beta * np.log(1 + u)) - 1) / (beta * u)

def F_kernel(r):
    """The F kernel function with two recognition lengths"""
    u1 = r / ell_1
    u2 = r / ell_2
    
    Xi1 = Xi_function(u1)
    Xi2 = Xi_function(u2)
    
    # Numerical derivative
    du = 1e-6
    Xi1_prime = (Xi_function(u1 + du) - Xi_function(u1 - du)) / (2 * du)
    Xi2_prime = (Xi_function(u2 + du) - Xi_function(u2 - du)) / (2 * du)
    
    F1 = Xi1 - u1 * Xi1_prime
    F2 = Xi2 - u2 * Xi2_prime
    
    return F1 + F2

def mu_interpolation(u):
    """MOND-like interpolation function"""
    return u / np.sqrt(1 + u**2)

def solve_information_field(r, v_bar):
    """Solve for the information field given baryonic velocities"""
    # Convert to SI units
    r_m = r * kpc_to_m
    v_bar_m = v_bar * km_to_m
    
    # Baryonic acceleration
    a_bar = v_bar_m**2 / r_m
    
    # Information field gradient (simplified approach)
    # In full implementation, would solve the differential equation
    # For now, use algebraic approximation
    
    # MOND-like interpolation parameter
    u = a_bar / g_dagger
    mu_val = mu_interpolation(u)
    
    # Information field contribution
    # This gives MOND-like behavior in deep field
    a_info = np.sqrt(a_bar * g_dagger) - a_bar
    
    # Apply kernel modulation
    F = F_kernel(r)
    a_info *= F
    
    return a_info

def predict_rotation_curve(galaxy_data):
    """Predict rotation curve using Recognition Science gravity"""
    r = galaxy_data['r']
    v_bar = galaxy_data['v_bar']
    
    # Solve for information field contribution
    a_info = solve_information_field(r, v_bar)
    
    # Convert back to velocity
    v_info = np.sqrt(np.abs(a_info * r * kpc_to_m)) / km_to_m
    
    # Total predicted velocity
    v_pred_squared = v_bar**2 + v_info**2
    v_pred = np.sqrt(v_pred_squared)
    
    return v_pred

def plot_galaxy(galaxy_data, v_pred):
    """Plot observed vs predicted rotation curve"""
    plt.figure(figsize=(10, 6))
    
    r = galaxy_data['r']
    v_obs = galaxy_data['v_obs']
    v_err = galaxy_data['v_err']
    v_bar = galaxy_data['v_bar']
    
    # Plot data
    plt.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Observed', markersize=5)
    plt.plot(r, v_bar, 'b--', label='Baryonic (Newtonian)', linewidth=2)
    plt.plot(r, v_pred, 'r-', label='RS Prediction', linewidth=2)
    
    plt.xlabel('Radius (kpc)', fontsize=12)
    plt.ylabel('Velocity (km/s)', fontsize=12)
    plt.title(f"{galaxy_data['name']} - Recognition Science Gravity Test", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate chi-squared
    mask = v_err > 0
    chi2 = np.sum(((v_obs[mask] - v_pred[mask]) / v_err[mask])**2)
    dof = np.sum(mask)
    chi2_reduced = chi2 / dof if dof > 0 else np.inf
    
    plt.text(0.95, 0.05, f'χ²/dof = {chi2_reduced:.2f}', 
             transform=plt.gca().transAxes, ha='right', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return chi2_reduced

def main():
    """Test Recognition Science gravity on multiple galaxies"""
    
    # Get list of galaxy files
    galaxy_files = glob.glob('Rotmod_LTG/*.dat')
    
    # Test on a few representative galaxies
    test_galaxies = ['NGC0300_rotmod.dat', 'NGC2403_rotmod.dat', 'NGC3198_rotmod.dat', 
                     'NGC6503_rotmod.dat', 'DDO154_rotmod.dat']
    
    chi2_values = []
    
    for galaxy_file in test_galaxies:
        full_path = f'Rotmod_LTG/{galaxy_file}'
        if os.path.exists(full_path):
            print(f"\nProcessing {galaxy_file}...")
            
            # Read data
            galaxy_data = read_galaxy_data(full_path)
            
            # Predict rotation curve
            v_pred = predict_rotation_curve(galaxy_data)
            
            # Plot and calculate chi-squared
            chi2_reduced = plot_galaxy(galaxy_data, v_pred)
            chi2_values.append(chi2_reduced)
            
            plt.savefig(f'RS_gravity_{galaxy_data["name"]}.png', dpi=150)
            plt.close()
            
            print(f"  χ²/dof = {chi2_reduced:.2f}")
    
    # Summary statistics
    if chi2_values:
        mean_chi2 = np.mean(chi2_values)
        median_chi2 = np.median(chi2_values)
        print(f"\nSummary:")
        print(f"  Mean χ²/dof = {mean_chi2:.2f}")
        print(f"  Median χ²/dof = {median_chi2:.2f}")
        print(f"  Recognition lengths: ℓ₁ = {ell_1} kpc, ℓ₂ = {ell_2} kpc")
        print(f"  No free parameters per galaxy!")

if __name__ == "__main__":
    main() 