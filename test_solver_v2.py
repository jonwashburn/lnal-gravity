#!/usr/bin/env python3
"""Test script for LNAL Advanced Solver V2"""

from lnal_advanced_solver_v2 import AdvancedLNALSolverV2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def main():
    print("Testing LNAL Advanced Solver V2...")
    solver = AdvancedLNALSolverV2()
    
    # Test on subset
    test_results = solver.solve_all_galaxies(max_galaxies=10)
    
    if test_results:
        print("\nTest completed successfully!")
        print(f"Mean χ²/N: {test_results['chi2_mean']:.2f}")
        print(f"Median χ²/N: {test_results['chi2_median']:.2f}")
        
        # Save plots
        solver.plot_best_examples()
        print("Plots saved to lnal_advanced_v2_best_examples.png")

if __name__ == "__main__":
    main() 