#!/usr/bin/env python3
"""
RS Gravity v5 - Parallel SPARC Analysis
Processes all 171 galaxies in parallel with optimized solvers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import json
from datetime import datetime
import time
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Import optimized solver
from rs_gravity_v5_optimized import RSGravityOptimized

# Physical constants
pc = 3.0857e16
kpc = 1000 * pc
M_sun = 1.989e30

print("=== RS Gravity v5 - Parallel SPARC Analysis ===")
print(f"CPUs available: {cpu_count()}")

def load_sparc_data():
    """Load SPARC galaxy data"""
    # Load main table
    sparc_file = 'SPARC_Lelli2016c.mrt.txt'
    if not os.path.exists(sparc_file):
        print(f"Error: {sparc_file} not found")
        return None
    
    # Read the data
    with open(sparc_file, 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('#') and len(line.split()) > 10:
            data_start = i
            break
    
    # Parse data
    galaxies = {}
    current_galaxy = None
    
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 11:
                try:
                    galaxy = parts[0]
                    r = float(parts[1]) * kpc  # Convert to meters
                    v_obs = float(parts[2]) * 1000  # Convert to m/s
                    err_v = float(parts[3]) * 1000
                    v_gas = float(parts[4]) * 1000
                    v_disk = float(parts[5]) * 1000
                    v_bulge = float(parts[6]) * 1000
                    
                    if galaxy not in galaxies:
                        galaxies[galaxy] = {
                            'r': [], 'v_obs': [], 'err_v': [],
                            'v_gas': [], 'v_disk': [], 'v_bulge': []
                        }
                    
                    galaxies[galaxy]['r'].append(r)
                    galaxies[galaxy]['v_obs'].append(v_obs)
                    galaxies[galaxy]['err_v'].append(err_v)
                    galaxies[galaxy]['v_gas'].append(v_gas)
                    galaxies[galaxy]['v_disk'].append(v_disk)
                    galaxies[galaxy]['v_bulge'].append(v_bulge)
                    
                except (ValueError, IndexError):
                    continue
    
    # Convert to numpy arrays
    for galaxy in galaxies:
        for key in galaxies[galaxy]:
            galaxies[galaxy][key] = np.array(galaxies[galaxy][key])
    
    print(f"Loaded {len(galaxies)} galaxies from SPARC")
    return galaxies

def estimate_density_profile(r, v_gas, v_disk, v_bulge):
    """Estimate baryon density from rotation curves"""
    # Total baryon velocity
    v_baryon_sq = v_gas**2 + v_disk**2 + v_bulge**2
    
    # Assume exponential disk
    h_R = np.median(r) / 3  # Scale length
    h_z = 300 * pc  # Scale height
    
    # Surface density from rotation curve
    # Σ ∝ v²/(GR) for exponential disk
    G_SI = 6.67430e-11
    Sigma = v_baryon_sq * h_R / (2 * np.pi * G_SI * r)
    
    # Volume density
    rho = Sigma / (2 * h_z)
    
    # Smooth and extrapolate
    mask = rho > 0
    if np.sum(mask) > 3:
        rho_smooth = interp1d(r[mask], rho[mask], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
        return rho_smooth(r)
    else:
        # Fallback: simple exponential
        rho_0 = 1e-21  # kg/m³
        return rho_0 * np.exp(-r / h_R)

def process_single_galaxy(galaxy_data, galaxy_name, params=None):
    """Process a single galaxy with RS gravity"""
    try:
        # Extract data
        r = galaxy_data['r']
        v_obs = galaxy_data['v_obs']
        err_v = galaxy_data['err_v']
        v_gas = galaxy_data['v_gas']
        v_disk = galaxy_data['v_disk']
        v_bulge = galaxy_data['v_bulge']
        
        # Quality cuts
        if len(r) < 5:
            return {'galaxy': galaxy_name, 'status': 'too_few_points'}
        
        # Estimate density
        rho_baryon = estimate_density_profile(r, v_gas, v_disk, v_bulge)
        
        # Create solver
        solver = RSGravityOptimized(galaxy_name, use_gpu=False)
        
        # Override parameters if provided
        if params:
            solver.beta = params.get('beta', solver.beta)
            solver.mu_0 = params.get('mu_0', solver.mu_0)
            solver.lambda_c = params.get('lambda_c', solver.lambda_c)
            solver.alpha_grad = params.get('alpha_grad', solver.alpha_grad)
        
        # Predict curve
        v_components = {'gas': v_gas, 'disk': v_disk, 'bulge': v_bulge}
        v_pred, v_baryon, t_elapsed = solver.predict_rotation_curve(r, rho_baryon, v_components)
        
        # Calculate chi-squared
        weights = 1.0 / (err_v**2 + (0.05 * v_obs)**2)  # 5% systematic error floor
        chi2 = np.sum(weights * (v_pred - v_obs)**2)
        chi2_per_n = chi2 / len(v_obs)
        
        # Additional metrics
        residuals = (v_pred - v_obs) / v_obs
        rms_percent = np.sqrt(np.mean(residuals**2)) * 100
        max_residual = np.max(np.abs(residuals)) * 100
        
        # Check for NaN
        if np.any(np.isnan(v_pred)):
            return {'galaxy': galaxy_name, 'status': 'nan_in_prediction'}
        
        return {
            'galaxy': galaxy_name,
            'status': 'success',
            'chi2_per_n': float(chi2_per_n),
            'rms_percent': float(rms_percent),
            'max_residual': float(max_residual),
            'n_points': len(r),
            'r_min_kpc': float(r[0] / kpc),
            'r_max_kpc': float(r[-1] / kpc),
            'v_max_kms': float(np.max(v_obs) / 1000),
            'time_ms': float(t_elapsed * 1000),
            'data': {
                'r': r.tolist(),
                'v_obs': v_obs.tolist(),
                'v_pred': v_pred.tolist(),
                'v_baryon': v_baryon.tolist()
            }
        }
        
    except Exception as e:
        return {
            'galaxy': galaxy_name,
            'status': 'error',
            'error': str(e)
        }

def parallel_process_sparc(galaxies, n_processes=None, params=None):
    """Process all SPARC galaxies in parallel"""
    if n_processes is None:
        n_processes = cpu_count() - 1  # Leave one CPU free
    
    print(f"\nProcessing {len(galaxies)} galaxies with {n_processes} processes...")
    
    # Prepare arguments
    galaxy_items = [(data, name) for name, data in galaxies.items()]
    
    # Create partial function with fixed params
    process_func = partial(process_single_galaxy, params=params)
    
    # Process in parallel
    t_start = time.time()
    
    with Pool(n_processes) as pool:
        # Use starmap for multiple arguments
        results = pool.starmap(process_func, galaxy_items)
    
    t_elapsed = time.time() - t_start
    
    print(f"\nCompleted in {t_elapsed:.1f} seconds")
    print(f"Average time per galaxy: {t_elapsed/len(galaxies)*1000:.1f} ms")
    
    return results

def analyze_results(results):
    """Analyze and summarize results"""
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)
    
    # Filter successful fits
    df_success = df[df['status'] == 'success'].copy()
    
    print(f"\n=== Analysis Summary ===")
    print(f"Total galaxies: {len(df)}")
    print(f"Successful fits: {len(df_success)} ({len(df_success)/len(df)*100:.1f}%)")
    
    if len(df_success) > 0:
        # Statistics
        print(f"\nχ²/N statistics:")
        print(f"  Median: {df_success['chi2_per_n'].median():.2f}")
        print(f"  Mean: {df_success['chi2_per_n'].mean():.2f}")
        print(f"  Std: {df_success['chi2_per_n'].std():.2f}")
        
        # Quality bins
        excellent = df_success[df_success['chi2_per_n'] < 1]
        good = df_success[(df_success['chi2_per_n'] >= 1) & (df_success['chi2_per_n'] < 5)]
        acceptable = df_success[(df_success['chi2_per_n'] >= 5) & (df_success['chi2_per_n'] < 10)]
        poor = df_success[df_success['chi2_per_n'] >= 10]
        
        print(f"\nQuality distribution:")
        print(f"  Excellent (χ²/N < 1): {len(excellent)} ({len(excellent)/len(df_success)*100:.1f}%)")
        print(f"  Good (1 ≤ χ²/N < 5): {len(good)} ({len(good)/len(df_success)*100:.1f}%)")
        print(f"  Acceptable (5 ≤ χ²/N < 10): {len(acceptable)} ({len(acceptable)/len(df_success)*100:.1f}%)")
        print(f"  Poor (χ²/N ≥ 10): {len(poor)} ({len(poor)/len(df_success)*100:.1f}%)")
        
        # Best and worst fits
        best = df_success.nsmallest(5, 'chi2_per_n')
        worst = df_success.nlargest(5, 'chi2_per_n')
        
        print(f"\nBest fits:")
        for _, row in best.iterrows():
            print(f"  {row['galaxy']}: χ²/N = {row['chi2_per_n']:.3f}")
        
        print(f"\nWorst fits:")
        for _, row in worst.iterrows():
            print(f"  {row['galaxy']}: χ²/N = {row['chi2_per_n']:.3f}")
    
    # Failed fits
    df_failed = df[df['status'] != 'success']
    if len(df_failed) > 0:
        print(f"\nFailed fits: {len(df_failed)}")
        print(df_failed['status'].value_counts())
    
    return df

def create_summary_plots(df_success, save_dir='sparc_v5_results'):
    """Create summary plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. χ²/N histogram
    ax = axes[0, 0]
    ax.hist(df_success['chi2_per_n'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df_success['chi2_per_n'].median(), color='red', linestyle='--', 
              label=f'Median = {df_success["chi2_per_n"].median():.2f}')
    ax.set_xlabel('χ²/N')
    ax.set_ylabel('Number of galaxies')
    ax.set_title('Fit Quality Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # 2. RMS vs χ²/N
    ax = axes[0, 1]
    ax.scatter(df_success['chi2_per_n'], df_success['rms_percent'], 
              alpha=0.6, s=30)
    ax.set_xlabel('χ²/N')
    ax.set_ylabel('RMS residual (%)')
    ax.set_title('RMS vs Fit Quality')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Performance
    ax = axes[0, 2]
    ax.scatter(df_success['n_points'], df_success['time_ms'],
              alpha=0.6, s=30)
    ax.set_xlabel('Number of data points')
    ax.set_ylabel('Computation time (ms)')
    ax.set_title('Performance Scaling')
    ax.grid(True, alpha=0.3)
    
    # 4. Galaxy properties
    ax = axes[1, 0]
    ax.scatter(df_success['r_max_kpc'], df_success['v_max_kms'],
              c=df_success['chi2_per_n'], cmap='viridis', 
              s=50, alpha=0.7, vmin=0, vmax=10)
    cb = plt.colorbar(ax.collections[0], ax=ax, label='χ²/N')
    ax.set_xlabel('R_max (kpc)')
    ax.set_ylabel('V_max (km/s)')
    ax.set_title('Galaxy Properties')
    ax.grid(True, alpha=0.3)
    
    # 5. Quality vs size
    ax = axes[1, 1]
    ax.scatter(df_success['r_max_kpc'], df_success['chi2_per_n'],
              alpha=0.6, s=30)
    ax.axhline(5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('R_max (kpc)')
    ax.set_ylabel('χ²/N')
    ax.set_title('Fit Quality vs Galaxy Size')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_text = f"""RS Gravity v5 - SPARC Analysis

Total galaxies: {len(df_success)}
Median χ²/N: {df_success['chi2_per_n'].median():.2f}
Mean χ²/N: {df_success['chi2_per_n'].mean():.2f}

Excellent fits (χ²/N < 1): {len(df_success[df_success['chi2_per_n'] < 1])}
Good fits (χ²/N < 5): {len(df_success[df_success['chi2_per_n'] < 5])}

Mean RMS: {df_success['rms_percent'].mean():.1f}%
Mean time: {df_success['time_ms'].mean():.1f} ms

Optimizations:
• Adaptive grids
• Batch operations
• Parallel processing"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.suptitle('RS Gravity v5 - Full SPARC Analysis', fontsize=16)
    plt.tight_layout()
    
    filename = os.path.join(save_dir, 'sparc_v5_summary.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot: {filename}")
    
    return filename

def save_results(results, df, save_dir='sparc_v5_results'):
    """Save all results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(save_dir, 'sparc_v5_raw_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary = {
        'version': 'v5_optimized',
        'timestamp': datetime.now().isoformat(),
        'n_galaxies': len(results),
        'n_success': len(df[df['status'] == 'success']),
        'statistics': {
            'median_chi2': float(df[df['status'] == 'success']['chi2_per_n'].median()),
            'mean_chi2': float(df[df['status'] == 'success']['chi2_per_n'].mean()),
            'std_chi2': float(df[df['status'] == 'success']['chi2_per_n'].std()),
            'mean_time_ms': float(df[df['status'] == 'success']['time_ms'].mean())
        }
    }
    
    with open(os.path.join(save_dir, 'sparc_v5_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save DataFrame as CSV
    df.to_csv(os.path.join(save_dir, 'sparc_v5_results.csv'), index=False)
    
    print(f"\nResults saved to {save_dir}/")

def main():
    """Run full SPARC analysis"""
    # Load data
    galaxies = load_sparc_data()
    if not galaxies:
        return
    
    # Process in parallel
    results = parallel_process_sparc(galaxies, n_processes=None)
    
    # Analyze
    df = analyze_results(results)
    df_success = df[df['status'] == 'success']
    
    # Create plots
    if len(df_success) > 0:
        create_summary_plots(df_success)
    
    # Save results
    save_results(results, df)
    
    print("\n=== SPARC Analysis Complete ===")
    print(f"Processed {len(galaxies)} galaxies")
    print(f"Success rate: {len(df_success)/len(df)*100:.1f}%")
    print(f"Median χ²/N: {df_success['chi2_per_n'].median():.2f}")

if __name__ == "__main__":
    main() 