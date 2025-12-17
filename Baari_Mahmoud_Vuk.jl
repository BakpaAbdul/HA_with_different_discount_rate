#####################################################################################################################
# Abdul Baari Bakpa , Vuk Rikovic, Mahmoud Tammam
#-----------------------------------------------------------------------------------------------------------------------------
#This is the solution code for Problem Set 5: Heterogeneous Agents - The Aiyagari Model with Preference Heterogeneity



##############################################################################################################################
## BLOCK 0 ##############################################################################################################################
##############################################################################################################################
using Distributions, LinearAlgebra, Plots

##Define Parameters
struct par_model
    beta::Float64              # discount factor
    mu::Float64            # risk aversion from CRRA parameter
    alpha::Float64            # capital share
    delta::Float64            # depreciation
    A::Float64            # aggregate productivity
    maxits::Int64
    tol::Float64
end

# --- Preference heterogeneity: two betas and type shares ---
const betaP   = 0.97          # patient type
const betaI   = 0.92          # impatient type
const lambdaP = 0.15           # share of patient households
const lambdaI = 1.0 - lambdaP # share of impatient households

# Parameter objects for each type (same technology, different beta)
parP = par_model(betaP, 2.0, 0.3, 0.03, 1.0, 3000, 1e-5)
parI = par_model(betaI, 2.0, 0.3, 0.03, 1.0, 3000, 1e-5)

##Defining the Grid for the Endogenous State Variable: Capital
Na = 200;
b = -0.2;
amax =  80;
agrid = collect(range(b, length = Na, stop = amax))

shift = 1.0 - b           # ensure positivity
agrid = (exp.(range(log(1), log(amax + shift), length=Na)) .- shift)


##Defining the Grid for the Exogenous State Variable: Technology Shock
rho = 0.9             # persistence of the AR(1) process
sigma = 0.2              # standard deviation of the AR(1) process
Ns = 5

function tauchen(mean, sd, rho, num_states; q=3)

    uncond_sd = sd/sqrt(1-rho^2)
    y = range(-q*uncond_sd, stop = q*uncond_sd, length = num_states)
    d = y[2]-y[1]

    Pi = zeros(num_states,num_states)

    for row = 1:num_states
      # end points
          Pi[row,1] = cdf(Normal(),(y[1] - rho*y[row] + d/2)/sd)
          Pi[row,num_states] = 1 - cdf(Normal(), (y[num_states] - rho*y[row] - d/2)/sd)

      # middle columns
          for col = 2:num_states-1
              Pi[row, col] = (cdf(Normal(),(y[col] - rho*y[row] + d/2) / sd) -
                             cdf(Normal(),(y[col] - rho*y[row] - d/2) / sd))
          end
    end

  yy = y .+ mean # center process around its mean

  Pi = Pi./sum(Pi, dims = 2) # renormalize

  return Pi, yy
end 

prob = tauchen(0, sigma, rho, Ns)[1];
logs = tauchen(0, sigma, rho, Ns)[2];
sgrid = exp.(logs);


##############################################################################################################################
## BLOCK 1 ##############################################################################################################################
##############################################################################################################################

VFI = function(r, w, agrid, sgrid, V0, prob, par)

    Ns = length(sgrid); Na = length(agrid); c = zeros(Na); U = zeros(Ns,Na,Na)     # Initialize the 3D Array

    ## Build the 3-Dimensional Contemporaneous Utility Grid for the System
    for is in 1:Ns                     # Loop Over skills Today
        for ia in 1:Na                 # Loop Over assets Today
            a = agrid[ia];     # Technology Today
            s = sgrid[is];    # Capital Tomorrow
            c .= (1+r)*a .+ s*w .- agrid                 # Solve for Consumption at Each Point
            U[is,ia,:] .= c.^(1-par.mu)./(1-par.mu);
            ii = findall(x -> x <= 0, c)
            U[is,ia,ii] .= -10^6;
    end
    end

    Vnew = copy(V0);  Vguess = copy(V0);  tv = zeros(Na);
    policy_a_index = Array{Int64,2}(undef,Ns,Na);

    for iter in 1:par.maxits
        for is in 1:Ns                     # Loop Over skills Today
            for ia in 1:Na                 # Loop Over assets Today
                tv = U[is,ia,:]' + par.beta*prob[is,:]'*Vguess[:,:]
                (Vnew[is,ia],policy_a_index[is,ia]) = findmax(tv[:])
            end
        end
        if maximum(abs,Vguess.-Vnew) < 1e-6
            println("Found solution after $iter iterations (VFI)")
            return policy_a_index, Vnew
        elseif iter==par.maxits
            println("No solution found after $iter iterations (VFI)")
         break
        end
        err = maximum(abs,Vguess.-Vnew)
        Vguess = copy(Vnew)  # update guess 
        # println("#iter = $iter, εᵥ = $err")
    end

end 




##############################################################################################################################
## BLOCK 2 ##############################################################################################################################
##############################################################################################################################

aiyagari = function(r, w, par, agrid, sgrid, prob, Vguess)

    Ns = length(sgrid); Na = length(agrid)

    policy_a_index, Vnew = VFI(r, w, agrid, sgrid, Vguess, prob, par)

    # Building the transition matrix 
    Q = zeros(Ns,Na,Ns,Na)
    for is in 1:Ns                     # Loop Over skills Today
        for ia in 1:Na                 # Loop Over assets Today
            for is_p in 1:Ns           # Loop Over skills Tomorrow
                ia_p = policy_a_index[is,ia]
                Q[is,ia,is_p,ia_p] = prob[is,is_p]
            end
        end
    end

    Q = reshape(Q, Ns*Na, Ns*Na)
    sum(Q, dims=2)     # Check that the rows sum to 1! 

    # Computing the stable distribution 
    dist = ones(1, Ns*Na) / (Ns*Na);
    dist = get_stable_dist(dist, Q,par)
    sum(dist)     # Check that the distribution vector sums to 1! 

    # Reshape dist as a Ns x Na dimension (more readable)
    dist = reshape(dist, Ns, Na)

    # Computing the aggregate
    agg_a = 0
    apol = agrid[policy_a_index]
    for is in 1:Ns                     # Loop Over skills Today
        for ia in 1:Na                 # Loop Over assets Today
            agg_a = apol[is,ia]*dist[is,ia] + agg_a
        end
    end

    return agg_a, Vnew, dist, apol
end 


function get_stable_dist(invdist, P, par)
    for iter in 1:par.maxits
        invdist2 = invdist * P
        if maximum(abs, invdist2 .- invdist) < par.tol
            println("Found solution after $iter iterations (dist)")
            return invdist2
        end
        invdist = invdist2
    end
    # If we reach here, we did not hit the tolerance but stop anyway
    println("Warning: get_stable_dist did not fully converge after $(par.maxits) iterations; returning last iterate")
    return invdist
end







##############################################################################################################################
## BLOCK 3 ##############################################################################################################################
##############################################################################################################################

# --- Bracket for r and initial guesses ---

minrate = -parP.delta
maxrate = (1 / min(parP.beta, parI.beta)) - 1.0   # upper bound from the more impatient type

VguessP = zeros(Ns, Na)   # initial guess for patient type value function
VguessI = zeros(Ns, Na)   # initial guess for impatient type value function

# --- Aggregate labor (same as before; beta doesn't matter for s-process) ---

labordist = ones(1, Ns) / Ns
labordist = get_stable_dist(labordist, prob, parP)   # use parP just for maxits/tol
agg_l = (labordist * sgrid)[1]   # Aggregate effective labor

# --- Market-clearing with two types ---

function find_eq(minrate, maxrate, parP, parI, lambdaP, agrid, sgrid, prob, VguessP, VguessI)

    for iter in 1:5000

        # 1. Candidate interest rate (midpoint of bracket)
        r0 = 0.5 * minrate + 0.5 * maxrate

        # 2. Firm side: capital demand K^d(r0) and wage w(r0)
        k0 = ((r0 + parP.delta) / (parP.alpha * agg_l^(1 - parP.alpha)))^(1 / (parP.alpha - 1))
        w  = (1 - parP.alpha) *
             ((parP.alpha * agg_l^(1 - parP.alpha)) / (r0 + parP.delta))^(parP.alpha / (1 - parP.alpha)) *
             agg_l^(-parP.alpha)

        # 3. Household side: capital supply of each type
        kP, VnewP, distP, apolP = aiyagari(r0, w, parP, agrid, sgrid, prob, VguessP)
        kI, VnewI, distI, apolI = aiyagari(r0, w, parI, agrid, sgrid, prob, VguessI)

        # Total capital supply: weighted by type shares
        k1 = lambdaP * kP + (1.0 - lambdaP) * kI

        # 4. Implied r from firm FOC given k1 (consistency check)
        r1 = parP.alpha * parP.A * max(0.001, k1)^(parP.alpha - 1) * agg_l^(1 - parP.alpha) - parP.delta
        err = r1 - r0
        testr = abs(err)

        println("iter = $iter, k0 = $k0, k1 = $k1, r0 = $r0, err = $err")

        # 5. Bisection update of r
        if testr > parP.tol
            if k1 > k0
                # supply too high -> r too high -> move upper bound down
                maxrate = r0
            else
                # supply too low -> r too low -> move lower bound up
                minrate = r0
            end
        else
            println("Equilibrium found")
            # optionally also construct total distribution/policy outside this function
            return k1, VnewP, VnewI, distP, distI, apolP, apolI, r1
        end

        # 6. Update value function guesses for next iteration
        VguessP = copy(VnewP)
        VguessI = copy(VnewI)
    end

    error("No equilibrium found after 5000 iterations")
end





# ------------------------------------
# Helpers for moments (add here)
# ------------------------------------

function wealth_marginals(agrid, dist_total)
    p_a = vec(sum(dist_total, dims = 1))
    p_a ./= sum(p_a)
    return p_a, agrid
end

function gini_from_groups(a, p)
    idx = sortperm(a)
    a_sorted = a[idx]
    p_sorted = p[idx]
    μ = sum(p_sorted .* a_sorted)
    μ == 0.0 && return 0.0
    G = 0.0
    n = length(a_sorted)
    for i in 1:n, j in 1:n
        G += p_sorted[i] * p_sorted[j] * abs(a_sorted[i] - a_sorted[j])
    end
    return G / (2.0 * μ)
end

function top_share_from_groups(a, p; q = 0.10)
    idx = sortperm(a)
    a_sorted = a[idx]
    p_sorted = p[idx]
    wealth = p_sorted .* a_sorted
    total_wealth = sum(wealth)
    total_wealth == 0.0 && return 0.0
    cum_pop = cumsum(p_sorted)
    cutoff = findfirst(x -> x ≥ 1.0 - q, cum_pop)
    cutoff = cutoff === nothing ? 1 : cutoff
    top_wealth = sum(wealth[cutoff:end])
    return top_wealth / total_wealth
end


# ------------------------------------
# Solve equilibrium and print summary
# ------------------------------------

k1, VnewP, VnewI, distP, distI, apolP, apolI, r1 =
    find_eq(minrate, maxrate, parP, parI, lambdaP, agrid, sgrid, prob, VguessP, VguessI)

dist_total = lambdaP * distP .+ (1.0 - lambdaP) * distI
p_a, a_levels = wealth_marginals(agrid, dist_total)
mean_wealth = sum(p_a .* a_levels)
gini_model  = gini_from_groups(a_levels, p_a)
top10_share = top_share_from_groups(a_levels, p_a; q = 0.10)

println("--------------------------------------------------")
println("Calibration summary for the two-type Aiyagari model")
println("--------------------------------------------------")
println("Preferences:")
println("  Patient type:   β_P = $(parP.beta), share λ_P = $(lambdaP)")
println("  Impatient type: β_I = $(parI.beta), share λ_I = $(1.0 - lambdaP)")
println()
println("Equilibrium prices and aggregates:")
println("  Equilibrium real interest rate r* = $(round(r1*100, digits=2))% (target ≈ 4%)")
println("  Aggregate capital K*              = $(round(k1, digits=3))")
println("  Mean wealth per household         = $(round(mean_wealth, digits=3))")
println()
println("Wealth distribution:")
println("  Model wealth Gini                 = $(round(gini_model, digits=3))  (target ≈ 0.75)")
println("  Model top  10% wealth share        = $(round(top10_share*100, digits=1))%  (target ≈ 75%)")
println("--------------------------------------------------")


# =======================
# Figures for the write-up
# =======================

# Figure 1: Savings policy functions by type at median productivity state
is_mid = Int(cld(Ns, 2))  # e.g. 3 if Ns = 5

plt1 = plot(
    agrid, apolP[is_mid, :],
    label = "Patient type",
    xlabel = "Current assets a",
    ylabel = "Next-period assets a'(a,s̄)",
    title = "Savings policy at median productivity state"
)
plot!(
    agrid, apolI[is_mid, :],
    label = "Impatient type"
)
savefig(plt1, "fig1_policy_functions.pdf")


# Figure 2: Wealth distribution by type (marginal over assets)
p_a_P = vec(sum(distP, dims = 1))
p_a_I = vec(sum(distI, dims = 1))
p_a_P ./= sum(p_a_P)
p_a_I ./= sum(p_a_I)

plt2 = plot(
    agrid, p_a_P,
    label = "Patient type",
    xlabel = "Assets a",
    ylabel = "Probability mass",
    title = "Wealth distribution by type"
)
plot!(
    agrid, p_a_I,
    label = "Impatient type"
)
savefig(plt2, "fig2_wealth_by_type.pdf")


# Figure 3: Lorenz curve of total wealth
wealth      = p_a .* a_levels
total_wealth = sum(wealth)

pop_cum    = cumsum(p_a)
wealth_cum = cumsum(wealth) ./ total_wealth

plt3 = plot(
    pop_cum, wealth_cum,
    label = "Lorenz curve",
    xlabel = "Cumulative population share",
    ylabel = "Cumulative wealth share",
    title = "Lorenz curve of wealth"
)
plot!([0.0, 1.0], [0.0, 1.0], linestyle = :dash, label = "45° line")

# Mark the bottom-90% point (top-10% share)
idx_90 = findfirst(x -> x ≥ 0.9, pop_cum)
if idx_90 !== nothing
    scatter!([pop_cum[idx_90]], [wealth_cum[idx_90]], label = "Bottom 90%")
end

savefig(plt3, "fig3_lorenz_curve.pdf")


# ==============================
# Figure: Capital demand vs supply
# ==============================

Nr = 25
r_min_plot = max(-parP.delta + 1e-4, 0.5 * r1)
r_max_plot = 1.5 * r1
r_grid = collect(range(r_min_plot, stop = r_max_plot, length = Nr))

Kd_vec = zeros(Nr)   # capital demand K^d(r)
Ks_vec = zeros(Nr)   # capital supply K^s(r)

for (i, r) in enumerate(r_grid)

    # --- Firm side: capital demand K^d(r) and wage w(r) ---
    Kd_vec[i] = ((r + parP.delta) /
                 (parP.alpha * agg_l^(1 - parP.alpha)))^(1 / (parP.alpha - 1))

    w = (1 - parP.alpha) *
        ((parP.alpha * agg_l^(1 - parP.alpha)) / (r + parP.delta))^(parP.alpha / (1 - parP.alpha)) *
        agg_l^(-parP.alpha)

    # --- Household side: capital supply of each type at this r ---
    # We don't reuse value functions here; just use VguessP/VguessI as in the main code
    kP, _, _, _ = aiyagari(r, w, parP, agrid, sgrid, prob, VguessP)
    kI, _, _, _ = aiyagari(r, w, parI, agrid, sgrid, prob, VguessI)

    Ks_vec[i] = lambdaP * kP + (1.0 - lambdaP) * kI
end

pltK = plot(
    r_grid .* 100, Kd_vec,
    label  = "Capital demand K^d(r)",
    xlabel = "Interest rate r (%)",
    ylabel = "Capital K",
    title  = "Capital demand and supply"
)
plot!(
    pltK, r_grid .* 100, Ks_vec,
    label = "Capital supply K^s(r)"
)
scatter!(
    pltK, [r1 * 100], [k1],
    label = "Equilibrium (r*, K*)"
)

savefig(pltK, "fig4_K_demand_supply.pdf")
