"""
    SingleComponentMetropolisHastings(proposal, x0, n, burnin, islog)

Passed to [`bayesianupdating`](@ref) to run the single-component Metropolis-Hastings algorithm starting from `x0` with univariate proposal distibution `proposal` (or vector of proposal distributions per dimension). Will generate `n` samples *after* performing `burnin` steps of the Markov chain and discarding the samples. The flag `islog` specifies whether the prior and likelihood functions passed to the [`bayesianupdating`](@ref) method are already given as logarithms.

Alternative constructor

```julia
    SingleComponentMetropolisHastings(proposal, x0, n, burnin)  # `islog` = true
```

"""
struct SingleComponentMetropolisHastings <: AbstractBayesianMethod
    proposal::Vector{<:UnivariateDistribution}
    x0::NamedTuple
    n::Int
    burnin::Int
    islog::Bool

    function SingleComponentMetropolisHastings(
            proposal::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}},
            x0::NamedTuple,
            n::Int,
            burnin::Int,
            islog::Bool = true,
        )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        if !isa(proposal, Vector)
            proposal = fill(proposal, length(x0))
        else
            @assert length(x0) == length(proposal) "Number of proposal distribution must match the dimension."
        end
        return new(proposal, x0, n, burnin, islog)
    end
end

"""
    bayesianupdating(prior, likelihood, models, mcmc)

Perform bayesian updating using the given `prior`, `likelihood`, `models`  and any MCMC sampler [`AbstractBayesianMethod`](@ref).

Alternatively the method can be called without `models`.

    bayesianupdating(prior, likelihood, mcmc)

When using [`TransitionalMarkovChainMonteCarlo`](@ref) the `prior` can automatically be constructed.

    bayesinupdating(likelihood, models, tmcmc)
    bayesianupdating(likelihood, tmcmc)


### Notes

`likelihood` is a Julia function which must be defined in terms of a `DataFrame` of samples, and must evaluate the likelihood for each row of the `DataFrame`

For example, a loglikelihood based on normal distribution using 'Data':

```julia
likelihood(df) = [sum(logpdf.(Normal.(df_i.x, 1), Data)) for df_i in eachrow(df)]
```

If a model evaluation is required to evaluate the likelihood, a vector of `UQModel`s must be passed to `bayesianupdating`. For example if the variable `x` above is the output of a numerical model.

"""
function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::Vector{<:UQModel},
        mh::SingleComponentMetropolisHastings,
    )
    number_of_dimensions = length(mh.x0)

    samples = DataFrame(collect(mh.x0)', collect(keys(mh.x0)))

    if !isempty(models)
        evaluate!(models, samples)
    end

    posterior = if mh.islog
        df -> likelihood(df) .+ prior(df)
    else
        df -> log.(likelihood(df)) .+ log.(prior(df))
    end

    rejection = 0.0

    for i in 1:(mh.n + mh.burnin - 1)
        current = DataFrame(samples[i, :])
        x = DataFrame(samples[i, :])

        for d in 1:number_of_dimensions
            x[1, d] += rand(mh.proposal[d])

            # safeguard for areas where the logprior is -Inf (prior = 0)
            while mh.islog ? isinf(prior(x)[1]) : isinf(log.(prior(x))[1])
                x[1, d] = current[1, d] + rand(mh.proposal[d])
            end

            if !isempty(models)
                evaluate!(models, x)
            end

            α = min(0, posterior(x)[1] - posterior(current)[1])

            if α >= log(rand())
                current[1, d] = x[1, d]
            else
                x[1, d] = current[1, d]
                rejection += 1
            end
        end

        push!(samples, x[1, :])
    end

    rejection /= ((mh.n + mh.burnin) * number_of_dimensions)

    # discard burnin samples during return
    return samples[(mh.burnin + 1):end, :], rejection
end

function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::UQModel,
        mh::SingleComponentMetropolisHastings,
    )
    return bayesianupdating(prior, likelihood, wrap(models), mh)
end

function bayesianupdating(
        prior::Function, likelihood::Function, mh::SingleComponentMetropolisHastings
    )
    return bayesianupdating(prior, likelihood, UQModel[], mh)
end

"""
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β, islog)

    Passed to [`bayesianupdating`](@ref) to run thetransitional Markov chain Monte Carlo algorithm  with [`RandomVariable'](@ref) vector `prior`. At each transitional level, one sample will be generated from `n` independent Markov chains after `burnin` steps have been discarded. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms.

Alternative constructors

```julia
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β)  # `islog` = true
    TransitionalMarkovChainMonteCarlo(prior, n, burnin)    # `β` = 0.2,  `islog` = true
```

# References

[chingTransitionalMarkovChain2007](@cite)

"""
struct TransitionalMarkovChainMonteCarlo <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    n::Int
    burnin::Int
    β::Real
    islog::Bool

    function TransitionalMarkovChainMonteCarlo(
            prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
            n::Int,
            burnin::Int,
            β::Real = 0.2,
            islog::Bool = true,
        )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        return new(prior, n, burnin, β, islog)
    end
end

# TMCMC implementation
function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::Vector{<:UQModel},
        tmcmc::TransitionalMarkovChainMonteCarlo,
    )
    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)

    rv_names = names(tmcmc.prior)
    n_rv = length(rv_names)

    j = 0 # iteration
    βⱼ = 0.0 # tempering

    θⱼ = sample(tmcmc.prior, tmcmc.n) # prior samples

    if !isempty(models)
        evaluate!(models, θⱼ)
    end

    S = 0.0

    while βⱼ < 1
        j += 1

        likelihood_j = tmcmc.islog ? likelihood(θⱼ) : log.(likelihood(θⱼ))

        adjust = maximum(likelihood_j)

        βⱼ⁺, wⱼ = _beta_and_weights(βⱼ, likelihood_j .- adjust)

        @debug "βⱼ" βⱼ⁺

        S += (log(mean(wⱼ)) + (βⱼ⁺ - βⱼ) * adjust)

        weights = FrequencyWeights(wⱼ ./ sum(wⱼ))

        idx = StatsBase.sample(collect(1:(tmcmc.n)), weights, tmcmc.n; replace = true)

        θⱼ⁺ = θⱼ[idx, :]

        Σⱼ = tmcmc.β^2 * cov(covariance_method, Matrix(θⱼ⁺[:, rv_names]))
        
        U = cholesky(Σⱼ).U   # factorize once for MvNormal sampler in MH part
            
        # Run inner MH algorithm

        chain = Vector{DataFrame}(undef, tmcmc.burnin + 2)

        chain[1] = copy(θⱼ⁺)

        target = if tmcmc.islog
            df -> likelihood(df) .* βⱼ⁺ .+ prior(df)
        else
            df -> log.(likelihood(df)) .* βⱼ⁺ .+ log.(prior(df))
        end

        for i in 2:(tmcmc.burnin + 2)
            next = copy(chain[i - 1])

            next[:, rv_names] = Matrix(next[:, rv_names]) .+ randn(tmcmc.n, n_rv) * U

            # safeguard for Inf in the prior
            idx_inf = findall(isinf, prior(next[:, rv_names]))

            while !isempty(idx_inf)
            
                means = Matrix(chain[i - 1][idx_inf, rv_names])
                next[idx_inf, rv_names] = means .+ randn(length(idx_inf), n_rv) * U
            
                still_inf = isinf.(prior(next[idx_inf, rv_names]))            # only re-check candidates
                idx_inf = idx_inf[still_inf]
            end

            if !isempty(models)
                evaluate!(models, next)
            end

            α = min.(0, target(next) .- target(chain[i - 1]))

            accept = α .>= log.(rand(length(α)))

            reject = .!accept

            next[reject, :] .= chain[i - 1][reject, :]

            chain[i] = next
        end

        θⱼ⁺ = chain[end]

        βⱼ = βⱼ⁺
        θⱼ = θⱼ⁺
    end

    model_calls = tmcmc.n * (1 + j * (1 + tmcmc.burnin))
    @debug "Model Calls" model_calls

    return θⱼ, S
end

function bayesianupdating(
        likelihood::Function,
        models::Vector{<:UQModel},
        tmcmc::TransitionalMarkovChainMonteCarlo,
    )
    prior = if tmcmc.islog
        df -> vec(
            sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims = 2),
        )
    else
        df -> vec(prod(hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims = 2))
    end

    return bayesianupdating(prior, likelihood, models, tmcmc)
end

function bayesianupdating(likelihood::Function, tmcmc::TransitionalMarkovChainMonteCarlo)
    return bayesianupdating(likelihood, UQModel[], tmcmc)
end

function bayesianupdating(
        prior::Function, likelihood::Function, tmcmc::TransitionalMarkovChainMonteCarlo
    )
    return bayesianupdating(prior, likelihood, UQModel[], tmcmc)
end

"""
    SequentialTransitionalMarkovChainMonteCarlo(prior, nrv_vec, particle_factor, burnin, β, islog)

    Passed to [`bayesianupdating`](@ref) to run the sequential version of the Transitional Markov Chain Monte Carlo algorithm with [`RandomVariable'](@ref) vector `prior`.
    The number of stages are implicitly defined via `length(nrv_vec)`.   
    The number of variables considered at each stage `s` are defined via `nrv_vec[s]` and will be `riors[1:nrv_vec[s]]`. 
    At each stage a single TMCMC run will be computed, considering previous TMCMC stage-runs as initialization. 
    At each transitional level, one sample will be generated from `n=nrv_vec[s]*particle_factor` independent Markov chains after `burnin` steps have been discarded. 
    Alternatively, `particle_factor` can be a vector having the same length as `nrv_vec`, and the numver of Markov chains will be `n=nrv_vec[s]*particle_factor[s]`. 
    The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already given as logarithms.

Alternative constructors

```julia
    SequentialTransitionalMarkovChainMonteCarlo(prior, nrv_vec, particle_factor, burnin, β)  # `islog` = true
    SequentialTransitionalMarkovChainMonteCarlo(prior, nrv_vec, particle_factor, burnin)    # `β` = 0.2,  `islog` = true
```

# References

[chingTransitionalMarkovChain2007](@cite)
[yangSequentialMarkovChainMonteCarlo2013](@cite)

"""
struct SequentialTransitionalMarkovChainMonteCarlo <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    nrv_vec::Vector{Int}
    particle_factor::Union{Int,Vector{Int}}
    burnin::Int
    β::Real
    islog::Bool

    function SequentialTransitionalMarkovChainMonteCarlo(
            prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
            nrv_vec::Vector{Int}, 
            particle_factor::Union{Int,Vector{Int}},
            burnin::Int,
            β::Real = 0.2,
            islog::Bool = true,
        )
        if any(nrv_vec .<= 0)
            error("Number of random variables per stage defined by `nrv_vec` must be positive")
        end

        if length(nrv_vec) > 1 && any(nrv_vec[1:end-1] .> nrv_vec[2:end])
            error("`nrv_vec` must be strictly monotonic increasing")
        end

        if nor(length(particle_factor) == 1, length(particle_factor) == length(nrv_vec))  
            error("`particle_factor` must be either length=1 or length(nrv_vec)")
        end

        if any(particle_factor .<= 0)
            error("Factor to compute number of particles `particle_factor` must be positive")
        end

        return new(prior, nrv_vec, particle_factor, burnin, β, islog)
    end
end

# sequential TMCMC implementation
function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::Vector{<:UQModel},
        stmcmc::SequentialTransitionalMarkovChainMonteCarlo,
    )
    # initialization
    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)
    S = 0.0    
    local_stage = 0
    model_calls = 0
    θⱼ = DataFrame()
    n_old = 0
        
    # outer loop over the number of random variables
    for (stage, n_rv) in enumerate(stmcmc.nrv_vec)

        @debug "Stage" stage

        # stage initialization
        rv_names = names(stmcmc.prior[1:n_rv])       # names of "active" rvs in this stage
        j = 0                                       # iteration counter in current stage
        
        # !TODO:
        # currently kind of heuristic approach to get number of samples - might be a better one 
        n_curr = length(stmcmc.particle_factor) == 1 ? n_rv*stmcmc.particle_factor : n_rv*stmcmc.particle_factor[stage]
            
        # for stage 1 additional initalization (same as TMCMC)
        if stage == 1
            
            βⱼ = 0.0                                    # tempering
            
            θⱼ = sample(stmcmc.prior[1:n_rv], n_curr)   # prior samples
            
            # eval forward model if supplied
            if !isempty(models)
                evaluate!(models, θⱼ)
            end
        
        else    # for stage >= 2 use previous results for initialization
            
            # compute likelihood from final coefficients of previous stage 
            #likelihood_old = stmcmc.islog ? likelihood(θⱼ) : log.(likelihood(θⱼ))
            
            # duplicate samples to match the new number of particles
            if mod(n_curr,n_old) == 0      # int-type duplication  
                k_tile = div(n_curr, n_old)
                idx_dup = repeat(1:n_old, k_tile)
            else    # uniform resampling
                idx_dup = StatsBase.sample(1:n_old, n_curr; replace=true)
            end
            
            # sample injection to match current number of particles
            θ_old = θⱼ[idx_dup, :]                      
            #likelihood_old = likelihood_old[idx_dup]
            
            # sample new RVs (added dimenions per particle) introduced in current stage
            θ_new = sample(stmcmc.prior[size(θ_old,2):n_rv], n_curr) 
            θ_curr = hcat(θ_old, θ_new)         # new samples with new RVs added
            
            # evaluate new likelihood
            if !isempty(models)
                evaluate!(models, θ_curr)
            end
            likelihood_curr = stmcmc.islog ? likelihood(θ_curr) : log.(likelihood(θ_curr))

            # compute initial β for current stage using the previous stage's likelihoods
            adjust_curr = Distributions.maximum(likelihood_curr)
            adjust_old  = 0.0 # Distributions.maximum(likelihood_old)
            L_curr = likelihood_curr .- adjust_curr
            #L_old  = likelihood_old  .- adjust_old
            
            # !TODO: 
            # The idea was here to compute an initial β > 0.0, to reflect the "informed" dimensions in the subsequent TMCMC
            # Sadly did not worked out and got β = eps() here, since COV(L_old) >> 1
            #
            # use modified version here since we need to get initial β (here γ) 
            # compute based on new weights are w = γ * L_curr - β_old * L_old (β_old = 1) 
            γ, wⱼ = _beta_and_weights(0.0, L_curr; extra = nothing)

            # resample based on the new weights to get the initial samples for the current stage
            weights = FrequencyWeights(wⱼ ./ sum(wⱼ))
            idx = StatsBase.sample(collect(1:n_curr), weights, n_curr; replace=true)
            
            # update model evidence, samples and β
            S += log(mean(wⱼ)) + γ * adjust_curr - adjust_old
            θⱼ = θ_curr[idx, :]
            βⱼ = γ

            @info("Finished initialization of stage $(stage), starting with β=$(βⱼ) and S=$(S)!")
        end

        # inner β loop
        while βⱼ < 1
            j += 1
            
            likelihood_j = stmcmc.islog ? likelihood(θⱼ) : log.(likelihood(θⱼ))

            adjust = Distributions.maximum(likelihood_j)

            βⱼ⁺, wⱼ = _beta_and_weights(βⱼ, likelihood_j .- adjust)

            @debug "βⱼ" βⱼ⁺

            S += (log(mean(wⱼ)) + (βⱼ⁺ - βⱼ) * adjust)
            
            weights = FrequencyWeights(wⱼ ./ sum(wⱼ))
            
            idx = StatsBase.sample(collect(1:(n_curr)), weights, n_curr; replace=true)
            
            θⱼ⁺ = θⱼ[idx, :]
            
            Σⱼ = stmcmc.β^2 * cov(covariance_method, Matrix(θⱼ⁺[:, rv_names]))

            U = cholesky(Σⱼ).U   # factorize once for MvNormal sampler in MH part

            # Run inner MH algorithm
            
            chain = Vector{DataFrame}(undef, stmcmc.burnin + 2)
            
            chain[1] = copy(θⱼ⁺)

            target = if stmcmc.islog
                df -> likelihood(df) .* βⱼ⁺ .+ prior(df[:,rv_names])
            else
                df -> log.(likelihood(df)) .* βⱼ⁺ .+ log.(prior(df[:,rv_names]))
            end

            for i in 2:(stmcmc.burnin + 2)
                next = copy(chain[i - 1])

                next[:, rv_names] = Matrix(next[:, rv_names]) .+ randn(n_curr, n_rv) * U

                # safeguard for Inf in the prior
                idx_inf = findall(isinf, prior(next[:, rv_names]))

                while !isempty(idx_inf)
                
                    means = Matrix(chain[i - 1][idx_inf, rv_names]) 
                    next[idx_inf, rv_names] = means .+ randn(length(idx_inf), n_rv) * U
                
                    still_inf = isinf.(prior(next[idx_inf, rv_names]))            # only re-check candidates
                    idx_inf = idx_inf[still_inf]
                end

                if !isempty(models)
                    evaluate!(models, next)
                end

                α = min.(0, target(next) .- target(chain[i - 1]))

                accept = α .>= log.(rand(length(α)))

                reject = .!accept

                next[reject, :] .= chain[i - 1][reject, :]

                chain[i] = next
            end

            θⱼ⁺ = chain[end]
            
            βⱼ = βⱼ⁺
            θⱼ = θⱼ⁺
            
        end # β loop

        model_calls += n_curr * (1 + j * (1 + stmcmc.burnin))
        n_old = n_curr
        
    end # stage loop

    @debug "Model Calls" model_calls

    return θⱼ, S
end

function bayesianupdating(
        likelihood::Function,
        models::Vector{<:UQModel},
        stmcmc::SequentialTransitionalMarkovChainMonteCarlo,
    )
    prior = if stmcmc.islog
        df -> vec(
            sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), stmcmc.prior[1:ncol(df)])...); dims = 2),
        )
    else
        df -> vec(prod(hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), stmcmc.prior[1:ncol(df)])...); dims = 2))
    end

    return bayesianupdating(prior, likelihood, models, stmcmc)
end

function bayesianupdating(likelihood::Function, stmcmc::SequentialTransitionalMarkovChainMonteCarlo)
    return bayesianupdating(likelihood, UQModel[], stmcmc)
end

function bayesianupdating(
        prior::Function, likelihood::Function, tmcmc::SequentialTransitionalMarkovChainMonteCarlo
    )
    return bayesianupdating(prior, likelihood, UQModel[], stmcmc)
end


# ---------------------------------------------------------------------------
## src
# Compute the next value for `β` and the nominal weights `w` using bisection.
    # Standard TMCMC stage (extra === nothing):
        #   w(x) = exp[ (x - β) * L ]
        #        = L_j(θ)^x / L_j(θ)^β        i.e. p(θ)/q(θ) with q,p sharing the SAME likelihood L_j
        #
    # Dimension-extension stage (extra = L_old, called with β = 0):
        #   w(γ) = eγp[ γ * L_curr - L_old ]
        #        = L_curr(θ)^γ / L_old(θ_old)^1
        #   i.e. w(γ) = p(γ|θ)/q(γ|θ) = L_curr(θ)^γ * π₀(θ) / ( L_old(θ_old)^{β_old=1} * π₀(θ) )
        #   with L_curr passed in as L, and extra = L_old accounting for
        #   the β_old = 1 enrichment already present in θ_old from the previous stage.

function _beta_and_weights(β::Real, L::AbstractVector{<:Real};
                            extra::Union{Nothing,AbstractVector{<:Real}} = nothing)
    low = β
    high = 2

    local x, w

    while (high - low) / middle(low, high) > 1e-6 && high > eps()
        x = middle(low, high)
        w = extra === nothing ? exp.((x - β) .* L) : exp.((x - β) .* L .- extra)

        if std(w) / mean(w) > 1
            high = x
        else
            low = x
        end
    end

    if x > 1
        x = 1
        w = extra === nothing ? exp.((x - β) .* L) : exp.((x - β) .* L .- extra)
    end

    return x, w
end
