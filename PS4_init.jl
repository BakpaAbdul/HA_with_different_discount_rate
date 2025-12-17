##############################################################################################################################
## BLOCK 0 ##############################################################################################################################
##############################################################################################################################
using Distributions, LinearAlgebra

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

par = par_model(0.95,2.0,0.3,0.03,1.0,3000,1e-6)

##Defining the Grid for the Endogenous State Variable: Capital
Na = 300;
b = -0.2;
amax =  60;
agrid = collect(range(b, length = Na, stop = amax));

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

function VFI(r, agrid, sgrid, V0, prob, par)

    Ns = length(sgrid)
    Na = length(agrid)

    # wage from firm's FOCs
    w = (1 - par.alpha) * (par.A * (par.alpha / (r + par.delta))^par.alpha)^(1 / (1 - par.alpha))

    # Precompute current-period utility u(c) for all (s, a, a')
    U = zeros(Ns, Na, Na)

    for is in 1:Ns                # productivity today
        for ia in 1:Na            # assets today
            for ia_p in 1:Na      # assets tomorrow
                a   = agrid[ia]       # assets today
                a_p = agrid[ia_p]     # assets tomorrow
                s   = sgrid[is]       # productivity today

                # Budget constraint: c = income today - savings
                c = (1 + r) * a + s * w - a_p

                if c < 0
                    U[is, ia, ia_p] = -1.0e6    # or -Inf
                else
                    U[is, ia, ia_p] = c^(1 - par.mu) / (1 - par.mu)
                end
            end
        end
    end

    # Initialize value function objects
    Vnew   = copy(V0)   # updated value function
    Vguess = copy(V0)   # current iterate of value function

    policy_a_index = Array{Int64,2}(undef, Ns, Na)
    tv = zeros(Na)      # temp: EV over s' for each a'

    ### VFI loop
    for iter in 1:par.maxits
        for is in 1:Ns
            # 1) Expected continuation value for each a' given current s
            @inbounds for ia_p in 1:Na
                tv[ia_p] = sum(prob[is, :] .* Vguess[:, ia_p])
            end

            # 2) For each current asset a, choose a' maximizing U + beta*EV
            @inbounds for ia in 1:Na
                Urow = vec(@view U[is, ia, :])      # length-Na vector over a'
                Q    = @. Urow + par.beta * tv      # Bellman RHS over choices a'
                val, idx = findmax(Q)               # idx is optimal a' index

                Vnew[is, ia]          = val
                policy_a_index[is, ia] = idx
            end
        end

        # 3) Convergence test
        if maximum(abs.(Vnew .- Vguess)) < par.tol
            println("VFI converged after $iter iterations")
            break
        end

        Vguess .= Vnew
    end

    return policy_a_index
end





##############################################################################################################################
## BLOCK 2 ##############################################################################################################################
##############################################################################################################################

aiyagari = function(r, par, agrid, sgrid, prob, Vguess)
    Ns = length(sgrid)
    Na = length(agrid)

    # 0. Households: get optimal policy a'(s,a) via VFI
    policy_a_index = VFI(r, agrid, sgrid, Vguess, prob, par)

    ##### 1. Building the transition matrix  ####################
    # Q[is, ia, js, ia_p] = P( s'=js, a'=ia_p | s=is, a=ia )
    Q = zeros(Ns, Na, Ns, Na)

    ### TO FILL ####### 
    # Compute Q 
    ##################
    for is in 1:Ns
        for ia in 1:Na
            ia_p = policy_a_index[is, ia]        # chosen next-period asset index
            for js in 1:Ns
                # probability of moving from (is, ia) to (js, ia_p)
                Q[is, ia, js, ia_p] = prob[is, js]
            end
        end
    end

    # Then reshape it if Q was 4D: ( (is,ia) , (js,ia_p) )
    Q = reshape(Q, Ns*Na, Ns*Na)

    # Check that the rows sum to 1! 
    row_sums = sum(Q, dims=2)
    println("Row sums:")
    println("  Min row sum = ", minimum(row_sums))
    println("  Max row sum = ", maximum(row_sums))

    ###### 2. Computing the stable distribution  ###################################
    dist = ones(1, Ns*Na) / (Ns*Na)
    dist = get_stable_dist(dist, Q, par)

    # Check that the distribution vector sums to 1! 
    sum(dist)

    ###### 3. Computing the aggregate #############################################
    # We want: agg_a = Σ_{s,a} dist(s,a) * a
    # reshape to Ns×Na for clarity
    dist_mat = reshape(dist, Ns, Na)

    agg_a = 0.0
    for is in 1:Ns
        for ia in 1:Na
            agg_a += dist_mat[is, ia] * agrid[ia]
        end
    end

    return agg_a
end





function get_stable_dist(invdist, P,par)
    for iter in 1:par.maxits
        invdist2 = invdist * P
        if maximum(abs, invdist2 .- invdist) < 1e-9
            println("Found solution after $iter iterations")
            return invdist2
        elseif iter == par.maxits
            error("No solution found after $iter iterations")
            return invdist
        end
        err = maximum(abs, invdist2 - invdist)
        invdist = invdist2
    end
end








##############################################################################################################################
## BLOCK 3 ##############################################################################################################################
##############################################################################################################################

## Intermediate, do not touch
# we just loop over 3 possible interest rates
r_vec = [0.01, 0.02, 0.05]
Vguess = zeros(Ns,Na)
agg_a = zeros(length(r_vec))

for (ir,r) in enumerate(r_vec)

    agg_a[ir] = aiyagari(r, par, agrid, sgrid,prob, Vguess) 

end 




 
# Firm capital demand K(r) from r + δ = α A K^{α-1}
Kd(r) = (par.alpha * par.A / (r + par.delta))^(1/(1 - par.alpha))
Kd_vec = [Kd(r) for r in r_vec]

using Plots
plot(r_vec, agg_a, label="Household asset supply A(r)", marker=:circle)
plot!(r_vec, Kd_vec, label="Firm capital demand K(r)", marker=:square,
      xlabel="Interest rate r", ylabel="Capital / Assets",
      title="Asset Supply and Capital Demand")
