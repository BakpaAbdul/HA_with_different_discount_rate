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

par = par_model(0.95,2.0,0.3,0.03,1.0,3000,1e-5)

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



function get_stable_dist(invdist, P,par)
    for iter in 1:par.maxits
        invdist2 = invdist * P
        if maximum(abs, invdist2 .- invdist) < 1e-9
            println("Found solution after $iter iterations (dist)")
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

minrate = -par.delta;
maxrate = (1/par.beta) - 1;
Vguess = zeros(Ns,Na)



minrate = -par.delta;
maxrate = (1/par.beta) - 1;
Vguess = zeros(Ns,Na)

labordist = ones(1,Ns)/Ns
labordist = get_stable_dist(labordist, prob,par)
agg_l = (labordist*sgrid)[1]; # Aggregate labor

function find_eq(minrate, maxrate, par, agrid, sgrid, prob, Vguess)
    iter = 0 
    for iter in 1:500
        r0 = 0.5*minrate + 0.5*maxrate
        k0 = ((r0+par.delta)/(par.alpha.*agg_l^(1-par.alpha)))^(1/(par.alpha-1)) # Demand 
        w = (1-par.alpha)*((par.alpha*agg_l^(1-par.alpha))/(r0 + par.delta))^(par.alpha/(1-par.alpha))*agg_l^(-par.alpha)

        k1, Vnew, dist, apol = aiyagari(r0, w, par, agrid, sgrid, prob, Vguess) # supplied
        
        r1 = par.alpha*par.A*max(0.001,k1)^(par.alpha-1)*agg_l^(1-par.alpha)-par.delta
        err=r1-r0;
        testr = abs(err)
        println("k0 = $k0, k1 = $k1, r = $r0, err = $err ($iter)")

        if testr > par.tol    
            if k1 > k0
                maxrate = r0;
            else
                minrate = r0;
            end
        else
            println("Equilibrium found")
            return(k1, Vnew, dist, apol, r1)
        end

        Vguess = copy(Vnew); 
        iter = iter + 1
    end

end 

k1, Vnew, dist, apol, r1 = find_eq(minrate, maxrate, par, agrid, sgrid, prob, Vguess)
plot(agrid,dist[:,:]')
plot(agrid,apol[:,:]')


