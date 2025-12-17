# Heterogeneous Agents with Different Discount Rates (Julia)

A compact Julia project studying a heterogeneous-agents (HA) macro model with differing discount rates across agent types. The repository includes scripts and figures that illustrate policy functions, wealth distribution by type, the Lorenz curve, and capital market equilibrium (demand vs. supply).

## Overview
- Goal: Explore how discount-rate heterogeneity affects savings behavior, wealth inequality, and aggregate capital.
- Outputs: Precomputed figures in `fig1_policy_functions.pdf`, `fig2_wealth_by_type.pdf`, `fig3_lorenz_curve.pdf`, and `fig4_K_demand_supply.pdf`.
- Scripts: Julia `.jl` files to set up parameters and run computations.

## Repo Contents
- `Baari_Mahmoud_Vuk.jl`: Main/primary Julia script for the HA model.
- `PS4_init.jl`: Initialization and helper routines (parameters, grids, etc.).
- `PS4 (1).jl`: Additional script with related computations or alternative setups.
- `fig*_*.pdf`: Generated figures summarizing model outcomes.
- `PS5.pdf`, `Baari_Vuk_Mahmoud.pdf`, `julia idea.pdf`: Write-ups/notes related to the assignment and approach.

## Quick Start
1. Install Julia if not already installed:
   - https://julialang.org/downloads/
2. Clone the repository:
   ```bash
   git clone https://github.com/BakpaAbdul/HA_with_different_discount_rate.git
   cd HA_with_different_discount_rate
   ```
3. Run the main script (adjust if you use a different entry file):
   ```bash
   julia Baari_Mahmoud_Vuk.jl
   ```

## Notes
- The current figures are included as PDFs. Re-running the scripts may regenerate them depending on the code paths.
- If you rename the repository (e.g., to `ha-discount-heterogeneity`), update any local paths and git remote URLs accordingly.

## License
- No license specified. Add one (e.g., MIT) if you plan to share or reuse the code broadly.
