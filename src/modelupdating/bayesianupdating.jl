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

        # creating zero-means MvNormal, since cov stays constant along MCMC chains
        mv_normal = MvNormal(zeros(n_rv), Σⱼ)

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

            next[:, rv_names] = Matrix(next[:, rv_names]) .+ transpose(rand(mv_normal, tmcmc.n))

            # safeguard for Inf in the prior
            idx_inf = findall(isinf, prior(next))

            while !isempty(idx_inf)
                current_means = Matrix(chain[i - 1][idx_inf, rv_names])

                next[idx_inf, rv_names] = current_means .+ transpose(rand(mv_normal, length(idx_inf)))

                # only re-checking idx_inf candidates
                still_inf = isinf.(prior(next[idx_inf, rv_names]))
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
    AbstractSurrogateSpec

Abstract type for surrogate specifications used in the transition step of [`SequentialTMCMC`](@ref).

Subtypes are:

- [`PCESurrogate`](@ref)
- [`ResponseSurfaceSurrogate`](@ref)
- [`RadialBasisSurrogate`](@ref)
"""
abstract type AbstractSurrogateSpec end

"""
    PCESurrogate(; p = 3, index_set = :TD)

Polynomial chaos expansion surrogate with Hermite basis (matched to standard normal priors), total degree `p` and multi-index set `index_set` (`:TD`, `:TP` or `:HC`). Used as default surrogate in [`SequentialTMCMC`](@ref).
"""
struct PCESurrogate <: AbstractSurrogateSpec
    p::Int
    index_set::Symbol
    function PCESurrogate(; p::Int = 3, index_set::Symbol = :TD)
        p > 0 || error("Polynomial degree `p` of `PCESurrogate` must be positive")
        index_set in (:TD, :TP, :HC) || error("`index_set` must be one of `:TD`, `:TP`, `:HC`")
        return new(p, index_set)
    end
end

"""
    ResponseSurfaceSurrogate(; p = 2)

Monomial response surface of degree `p`, see [`ResponseSurface`](@ref). Can be used as surrogate in [`SequentialTMCMC`](@ref).
"""
struct ResponseSurfaceSurrogate <: AbstractSurrogateSpec
    p::Int
    function ResponseSurfaceSurrogate(; p::Int = 2)
        p > 0 || error("Polynomial degree `p` of `ResponseSurfaceSurrogate` must be positive")
        return new(p)
    end
end

"""
    RadialBasisSurrogate(; ε = 1.0)

Gaussian radial basis function surrogate with shape parameter `ε`, centered at the training points. Can be used as surrogate in [`SequentialTMCMC`](@ref).
"""
struct RadialBasisSurrogate <: AbstractSurrogateSpec
    ε::Float64
    RadialBasisSurrogate(; ε::Float64 = 1.0) = new(ε)
end

"""
    AbstractTransitionMode

Abstract type for the transition mode used in stages `s >= 2` of [`SequentialTMCMC`](@ref).

Subtypes are:

- [`SurrogateTransition`](@ref) (default)
- [`FreezeUnfreezeTransition`](@ref)
"""
abstract type AbstractTransitionMode end

"""
    SurrogateTransition(; surrogate = PCESurrogate(), surrogate_steps = 5)

Default transition of [`SequentialTMCMC`](@ref): weighted resampling of the previous stage's particles, prior injection of the new dimensions, per-stage surrogate refinement of the particles, and an initial `β > 0` computed from the incremental likelihood.
"""
struct SurrogateTransition <: AbstractTransitionMode
    surrogate::AbstractSurrogateSpec
    surrogate_steps::Int
    function SurrogateTransition(;
            surrogate::AbstractSurrogateSpec = PCESurrogate(), surrogate_steps::Int = 5
        )
        surrogate_steps >= 0 ||
            error("Number of surrogate refinement steps `surrogate_steps` must be non-negative")
        return new(surrogate, surrogate_steps)
    end
end

"""
    FreezeUnfreezeTransition(; ess_threshold = 0.5)

Alternative transition of [`SequentialTMCMC`](@ref): after growing the particle dimension, the inherited (previously fitted) dimensions are frozen in the MH proposal while the effective sample size (ESS) of the incremental weights is below `ess_threshold * n_curr`; once the ESS stabilizes above the threshold, all active dimensions are unfrozen. Only one likelihood evaluation per MH step is needed.
"""
struct FreezeUnfreezeTransition <: AbstractTransitionMode
    ess_threshold::Float64
    function FreezeUnfreezeTransition(; ess_threshold::Float64 = 0.5)
        0 < ess_threshold <= 1 || error("`ess_threshold` must be in (0, 1]")
        return new(ess_threshold)
    end
end

function _ess(w::AbstractVector{<:Real})
    p = w ./ sum(w)
    return 1 / sum(p .^ 2)
end

function _surrogate_input_names(d::Int)
    return [Symbol("ξ$i") for i in 1:d]
end

function _fit_surrogate(spec::PCESurrogate, X::Matrix{<:Real}, y::Vector{<:Real})
    d = size(X, 2)
    Ψ = PolynomialChaosBasis(fill(HermiteBasis(), d), spec.p, spec.index_set)
    A = Matrix{Float64}(undef, size(X, 1), length(Ψ.α))
    for i in axes(X, 1)
        A[i, :] .= evaluate(Ψ, view(X, i, :))
    end
    coef = A \ y
    return ξ -> dot(evaluate(Ψ, ξ), coef)
end

function _fit_surrogate(spec::ResponseSurfaceSurrogate, X::Matrix{<:Real}, y::Vector{<:Real})
    inputs = _surrogate_input_names(size(X, 2))
    df = DataFrame(X, inputs)
    df.L = y
    rs = ResponseSurface(df, :L, spec.p, inputs)
    return ξ -> begin
        d = DataFrame(reshape(collect(ξ), 1, :), inputs)
        evaluate!(rs, d)
        return d.L[1]
    end
end

function _fit_surrogate(spec::RadialBasisSurrogate, X::Matrix{<:Real}, y::Vector{<:Real})
    inputs = _surrogate_input_names(size(X, 2))
    df = DataFrame(X, inputs)
    df.L = y
    rbf = LinearBasisFunctionModel(df, :L, GaussianRadialBasis(copy(X'), spec.ε), inputs)
    return ξ -> begin
        d = DataFrame(reshape(collect(ξ), 1, :), inputs)
        evaluate!(rbf, d)
        return d.L[1]
    end
end

function _surrogate_refine(
        prior::Function, Lhat::Function, θ0::DataFrame, rv_names::Vector{Symbol}, n_steps::Int
    )
    n = nrow(θ0)
    d = length(rv_names)
    θ = copy(θ0)
    sd = max.(vec(std(Matrix(θ[:, rv_names]); dims = 1)), 1e-3)
    logtarget(df) = prior(df[:, rv_names]) .+ [Lhat(row) for row in eachrow(Matrix(df[:, rv_names]))]
    cur = logtarget(θ)
    for _ in 1:n_steps
        prop = copy(θ)
        prop[:, rv_names] = Matrix(prop[:, rv_names]) .+ randn(n, d) .* sd'
        lp = logtarget(prop)
        accept = min.(0.0, lp .- cur) .>= log.(rand(n))
        Xc = Matrix(θ[:, rv_names])
        Xc[accept, :] = Matrix(prop[:, rv_names])[accept, :]
        θ[:, rv_names] = Xc
        cur[accept] .= lp[accept]
    end
    return θ
end

"""
    SequentialTMCMC(prior, nrv_vec, particle_factor, burnin, β, islog; transition = SurrogateTransition())

    Passed to [`bayesianupdating`](@ref) to run a sequential (dimension-growing) version of the Transitional Markov Chain Monte Carlo algorithm with [`RandomVariable`](@ref) vector `prior`.
    The number of stages are implicitly defined via `length(nrv_vec)`.
    The number of variables considered at each stage `s` are defined via `nrv_vec[s]` and will be `prior[1:nrv_vec[s]]`.
    At each stage a single TMCMC run will be computed, considering the previous stage's run as initialization.
    At each stage, one sample will be generated from `n=nrv_vec[s]*particle_factor` independent Markov chains after `burnin` steps have been discarded.
    Alternatively, `particle_factor` can be a vector having the same length as `nrv_vec`, and the number of Markov chains will be `n=nrv_vec[s]*particle_factor[s]`.
    The flag `islog` specifies whether the prior and likelihood functions passed to the [`bayesianupdating`](@ref) method are already given as logarithms.

    For stages `s >= 2` the transition from the previous stage is controlled by the keyword `transition`:
    - [`SurrogateTransition`](@ref) (default): the previous stage's particles are resampled according to their final (posterior) weights, the newly added dimensions are initialized from the prior and evaluated once with the forward model, a surrogate of the log-likelihood is fitted per stage to the resulting samples and used to refine the particles with a short surrogate-only Markov chain (no forward model evaluations), and finally an initial tempering value `β > 0` is computed from the incremental likelihood `L_curr - L_old` via bisection.
    - [`FreezeUnfreezeTransition`](@ref): the inherited dimensions are frozen in the MH proposal until the ESS of the incremental weights stabilizes, then all dimensions are unfrozen. Requires only one likelihood evaluation per MH step.

Alternative constructors

```julia
    SequentialTMCMC(prior, nrv_vec, particle_factor, burnin, β)                    # `islog` = true
    SequentialTMCMC(prior, nrv_vec, particle_factor, burnin)                        # `β` = 0.2, `islog` = true
    SequentialTMCMC(prior, nrv_vec, particle_factor, burnin; transition = SurrogateTransition(surrogate = PCESurrogate(p = 4)))
    SequentialTMCMC(prior, nrv_vec, particle_factor, burnin; transition = FreezeUnfreezeTransition(ess_threshold = 0.5))
```

# References

[chingTransitionalMarkovChain2007](@cite)
[yangSequentialMarkovChainMonteCarlo2013](@cite)

"""
struct SequentialTMCMC <: AbstractBayesianMethod
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    nrv_vec::Vector{Int}
    particle_factor::Union{Int,Vector{Int}}
    burnin::Int
    β::Real
    islog::Bool
    transition::AbstractTransitionMode

    function SequentialTMCMC(
            prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
            nrv_vec::Vector{Int},
            particle_factor::Union{Int,Vector{Int}},
            burnin::Int,
            β::Real = 0.2,
            islog::Bool = true;
            transition::AbstractTransitionMode = SurrogateTransition(),
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

        return new(prior, nrv_vec, particle_factor, burnin, β, islog, transition)
    end
end

# sequential (dimension-growing) TMCMC implementation
function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::Vector{<:UQModel},
        stmcmc::SequentialTMCMC,
    )
    # initialization
    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)
    S = 0.0
    model_calls = 0
    θ* = DataFrame()
    L* = Vector{Float64}()
    n_old = 0    # number of particles of the previous stage
    d_old = 0    # number of random variables of the previous stage

    # outer loop over the stages (increasing number of random variables)
    for (stage, n_rv) in enumerate(stmcmc.nrv_vec)

        @debug "Stage" stage

        # stage initialization
        rv_names = names(stmcmc.prior[1:n_rv])       # names of "active" rvs in this stage
        j = 0                                       # iteration counter in current stage

        n_curr = length(stmcmc.particle_factor) == 1 ? n_rv*stmcmc.particle_factor : n_rv*stmcmc.particle_factor[stage]

        # for stage 1 additional initialization (same as TMCMC)
        if stage == 1

            β* = 0.0                                    # tempering

            θ* = sample(stmcmc.prior[1:n_rv], n_curr)   # prior samples

            # eval forward model if supplied
            if !isempty(models)
                evaluate!(models, θ*)
            end

        else    # for stage >= 2 use previous results for initialization

            # 1. weighted resampling of the previous stage's particles using their
            #    final (beta = 1) posterior weights - no new likelihood evaluations
            w_old = exp.(L* .- maximum(L*))
            idx_resample = StatsBase.sample(collect(1:n_old), FrequencyWeights(w_old), n_curr; replace = true)
            θ_old = θ*[idx_resample, :]
            L_old = L*[idx_resample]

            # 2. initialize the newly added dimensions from the prior and evaluate once
            θ_new = sample(stmcmc.prior[d_old+1:n_rv], n_curr)
            θ_curr = hcat(θ_old, θ_new)

            if !isempty(models)
                evaluate!(models, θ_curr)
            end
            L_curr = stmcmc.islog ? likelihood(θ_curr) : log.(likelihood(θ_curr))
            model_calls += n_curr

            if stmcmc.transition isa SurrogateTransition

                # 3. fit a per-stage surrogate of the log-likelihood and refine the
                #    particles with a short surrogate-only Markov chain (no forward solves)
                if stmcmc.transition.surrogate_steps > 0
                    Lhat = _fit_surrogate(
                        stmcmc.transition.surrogate, Matrix(θ_curr[:, rv_names]), L_curr
                    )
                    θ_curr = _surrogate_refine(
                        prior, Lhat, θ_curr, rv_names, stmcmc.transition.surrogate_steps
                    )
                    if !isempty(models)
                        evaluate!(models, θ_curr)
                    end
                    L_curr = stmcmc.islog ? likelihood(θ_curr) : log.(likelihood(θ_curr))
                    model_calls += n_curr
                end

                # 4. compute an initial beta > 0 for the current stage from the
                #    incremental likelihood L_curr - L_old (previous stage at beta = 1)
                adjust_curr = maximum(L_curr)
                adjust_old = maximum(L_old)
                γ, w* = _beta_and_weights(0.0, L_curr .- adjust_curr; extra = L_old .- adjust_old)

                # resample based on the new weights to get the initial samples for the current stage
                S += log(mean(w*)) + γ * adjust_curr - adjust_old
                idx = StatsBase.sample(collect(1:n_curr), FrequencyWeights(w* ./ sum(w*)), n_curr; replace = true)

                θ* = θ_curr[idx, :]
                L* = L_curr[idx]
                β* = γ

            else    # FreezeUnfreezeTransition: start from beta = 0, freeze old dims until ESS stabilizes

                L* = L_curr
                θ* = θ_curr
                β* = 0.0

            end

            @info("Finished initialization of stage $(stage), starting with β=$(β*) and S=$(S)!")
        end

        # inner β loop
        while β* < 1
            j += 1

            likelihood_j = stmcmc.islog ? likelihood(θ*) : log.(likelihood(θ*))

            adjust = Distributions.maximum(likelihood_j)

            β*⁺, w* = _beta_and_weights(β*, likelihood_j .- adjust)

            @debug "β*" β*⁺

            S += (log(mean(w*)) + (β*⁺ - β*) * adjust)

            weights = FrequencyWeights(w* ./ sum(w*))

            idx = StatsBase.sample(collect(1:(n_curr)), weights, n_curr; replace=true)

            θ*⁺ = θ*[idx, :]

            # freeze/unfreeze decision based on the ESS of the resampling weights:
            # while the ESS is low (new dimensions not yet annealed), the MH proposal
            # only moves the newly added dimensions; once the ESS stabilizes above
            # the threshold, all active dimensions are unfrozen
            frozen = false
            if stage > 1 && stmcmc.transition isa FreezeUnfreezeTransition
                frozen = _ess(w*) < stmcmc.transition.ess_threshold * n_curr
            end
            prop_names = frozen ? rv_names[d_old+1:end] : rv_names

            Σ* = stmcmc.β^2 * cov(covariance_method, Matrix(θ*⁺[:, prop_names]))

            U = cholesky(Σ*).U   # factorize once for MvNormal sampler in MH part

            # Run inner MH algorithm

            chain = Vector{DataFrame}(undef, stmcmc.burnin + 2)

            chain[1] = copy(θ*⁺)

            target = if stmcmc.islog
                df -> likelihood(df) .* β*⁺ .+ prior(df[:,rv_names])
            else
                df -> log.(likelihood(df)) .* β*⁺ .+ log.(prior(df[:,rv_names]))
            end

            for i in 2:(stmcmc.burnin + 2)
                next = copy(chain[i - 1])

                next[:, prop_names] = Matrix(next[:, prop_names]) .+ randn(n_curr, length(prop_names)) * U

                # safeguard for Inf in the prior
                idx_inf = findall(isinf, prior(next[:, rv_names]))

                while !isempty(idx_inf)

                    means = Matrix(chain[i - 1][idx_inf, prop_names])
                    next[idx_inf, prop_names] = means .+ randn(length(idx_inf), length(prop_names)) * U

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

            θ*⁺ = chain[end]

            β* = β*⁺
            θ* = θ*⁺

        end # β loop

        # store final likelihoods of the stage for the weighted resampling of the next stage
        L* = stmcmc.islog ? likelihood(θ*) : log.(likelihood(θ*))

        model_calls += n_curr * (1 + j * (1 + stmcmc.burnin))
        n_old = n_curr
        d_old = n_rv

    end # stage loop

    @debug "Model Calls" model_calls

    return θ*, S
end

function bayesianupdating(
        likelihood::Function,
        models::Vector{<:UQModel},
        stmcmc::SequentialTMCMC,
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

function bayesianupdating(likelihood::Function, stmcmc::SequentialTMCMC)
    return bayesianupdating(likelihood, UQModel[], stmcmc)
end

function bayesianupdating(
        prior::Function, likelihood::Function, tmcmc::SequentialTMCMC
    )
    return bayesianupdating(prior, likelihood, UQModel[], tmcmc)
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
