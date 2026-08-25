## Reconstruction of diffusion field for 2D advection diffusion problem
using UncertaintyQuantification
using Random
using LinearAlgebra
import GaussianRandomFields as GRF
using SparseArrays
using DataFrames
using Plots
import StatsPlots

Random.seed!(13)
ENV["JULIA_DEBUG"] = "UncertaintyQuantification"

## advection-diffusion problem
    # code below assumes 2D square domain [0,hx*nx]×[0,hy*ny]
    # fixed periodic BC in x and y
struct AdvDiffProb
    nx :: Int
    ny :: Int
    dx :: Real
    dy :: Real
    ∇x :: SparseMatrixCSC{Float64,Int}
    ∇y :: SparseMatrixCSC{Float64,Int}
    Δ  :: SparseMatrixCSC{Float64,Int}
end
    # build matrices
function build_1d_matrices(n::Int, h::Real)
    # 4th-order central: 
        #  ∂/∂x  stencil [ 1, -8,   0,  8, -1) / (12h)
        # ∂²/∂x² stencil [-1, 16, -30, 16, -1) / (12h²)
    #                     
    ∂1 = spzeros(n, n)
    ∂2 = spzeros(n, n)
    for i in 1:n
        # periodic BC via mod1
        im2, im1, ip1, ip2 = mod1.([i-2,i-1,i+1,i+2],n)
        ∂1[i, im2] +=  1 / (12h)
        ∂1[i, im1] -=  8 / (12h)
        ∂1[i, ip1] +=  8 / (12h)
        ∂1[i, ip2] -=  1 / (12h)
        ∂2[i, im2] -=  1 / (12h^2)
        ∂2[i, im1] += 16 / (12h^2)
        ∂2[i, i]   -= 30 / (12h^2)
        ∂2[i, ip1] += 16 / (12h^2)
        ∂2[i, ip2] -=  1 / (12h^2)
    end
    return ∂1, ∂2  
end

function build_prob(nx::Int, ny::Int, hx::Real, hy::Real)
    ∂1x, ∂2x = build_1d_matrices(nx, hx)
    ∂1y, ∂2y = build_1d_matrices(ny, hy)
    Ix, Iy = sparse(I, nx, nx), sparse(I, ny, ny)

    ∇x = kron(Iy, ∂1x)     # full 2D grid ∂x matrix
    ∇y = kron(∂1y, Ix)     # full 2D grid ∂y matrix
    Δ  = kron(Iy, ∂2x) + kron(∂2y, Ix)   # Laplacian

    return AdvDiffProb(nx, ny, hx, hy, ∇x, ∇y, Δ)
end
    # assemble discrete advection and diffusion operators
function advection_operator(vx::AbstractVector{<:Real}, vy::AbstractVector{<:Real}, adp::AdvDiffProb)
    return Diagonal(vx) * adp.∇x + Diagonal(vy) * adp.∇y
end

function advection_operator(vx::Real, vy::Real, adp::AdvDiffProb)
    return vx .* adp.∇x + vy .* adp.∇y
end

function diffusion_operator(d::AbstractVector{<:Real}, adp::AdvDiffProb)
    dDdx = adp.∇x * d
    dDdy = adp.∇y * d
    return Diagonal(d) * adp.Δ + Diagonal(dDdx) * adp.∇x + Diagonal(dDdy) * adp.∇y
end
    # build grid
function build_grid(nx::Int, ny::Int, hx::Real, hy::Real)
    Lx, Ly = (nx-1)*hx, (ny-1)*hy
    xvec = range(0, Lx, length=nx)
    yvec = range(0, Ly, length=ny)
    xmat = [x for x in xvec, y in yvec]
    ymat = [y for x in xvec, y in yvec]
    return xvec, yvec, Lx, Ly, vec(xmat), vec(ymat)
end

function build_u0(xvec::AbstractVector{<:Real},yvec::AbstractVector{<:Real},
                    kx::Int,ky::Int,Lx::Real,Ly::Real;type::String="sin")
    if type == "sin"
        return sin.(π*kx .* xvec ./ Lx) .* sin.(π*ky .* yvec ./ Ly) .+ 1.0
    elseif type == "exp"
        return 2.0 .* exp.(-((xvec .- Lx/2).^2 .+ (yvec .- Ly/2).^2) ./ (2*min(xvec[3],yvec[3])^2))
    end
end
    # IMEX Crank-Nicolson solve
    # (I − Δt/2 ​DiffOp(d_vec))*u_n+1 = (I + Δt/2 ​DiffOp(d_vec))*u_n − Δt a_mat*u
function forward_solve_imex(d::AbstractVector, u0::AbstractVector,
                             vx::Real, vy::Real, 
                             adp::AdvDiffProb, 
                             dt::Real, nsteps::Int, n_out::Int)      
    # CFL check
    @assert dt <= min(adp.dx/abs(vx),adp.dy/abs(vy)) "dt should be smaller than $(min(adp.dx*abs(vx),adp.dy*abs(vy)))"
    
    # get operator-matrices                         
    Dmat = diffusion_operator(d, adp)
    Amat = advection_operator(vx, vy, adp)

    LHS = I - (dt/2) .* Dmat
    RHSmat = I + (dt/2) .* Dmat
    F = lu(LHS)    # factorize once per diffusion-vector and reuse at each timestep

    u = copy(u0)
    uout = spzeros(adp.nx*adp.ny,n_out+1)
    uout[:,1] = u0
    idx_out = Int.(round.(LinRange(0,nsteps,n_out+1),digits=0))[2:end]
    j = 1
    for step in 1:nsteps
        rhs = u .- dt .* (Amat * u)
        u = F \ rhs
        if step in idx_out
            j += 1
            uout[:,j] = u
        end
    end
    return uout
end

## parameters
nx, ny = 51, 31     # points per dim
hx, hy = 0.2, 0.2   # grid size per dim
vx, vy = 1.0, 0.5   # sclar velocity per dim
kx, ky = 2, 3       # u0 half sin-frequency per dim, e.g. sin.(π*kx .* xvec ./ Lx)
dt     = 0.01       # time step size
nsteps = 500        # number of time steps 
n_out  = 5          # save only (n_out+1) equidistant frames (+1 is the IC)   
T_reco = 30         # Truncation order for reconstruction 
T_ref  = 50         # Truncation order for reference solution
σ_noise = 0.05      # Noise std

## create reference solution
xunique, yunique, Lx, Ly, xvec, yvec = build_grid(nx, ny, hx, hy) 
adp = build_prob(nx, ny, hx, hy)

u0 = build_u0(xvec,yvec,kx,ky,Lx,Ly;type="sin")
u0 = build_u0(xvec,yvec,kx,ky,Lx,Ly;type="sin")

cov = GRF.CovarianceFunction(2, GRF.Matern(5/4, 3/4)) 
grf = GRF.GaussianRandomField(cov, GRF.KarhunenLoeve(T_ref), xunique, yunique)
totalE = sum(grf.data.eigenval.^2) ./ (1-GRF.rel_error(grf))    # should be Lx*Ly
sample_d(samp;λ=0.1) = λ .* exp.(@views grf.data.eigenfunc[:, eachindex(samp)] * (grf.data.eigenval[eachindex(samp)] .* samp))


# per particle:
s_ref = randn(T_ref)
d_ref = sample_d(s_ref)
heatmap(xunique,yunique,reshape(d_ref,(nx,ny))',title="Reference Diffusion field")

u_ref = forward_solve_imex(d_ref, u0, vx, vy, adp, dt, nsteps, n_out)
p = []
for i = 1:size(u_ref,2)
    f = heatmap(xunique,yunique,reshape(u_ref[:,i],(nx,ny))',clims=(0,2),title="Frame$(i)")
    plot!(f,xlims=(0,Lx),ylims=(0,Ly))
    push!(p,f)
end
plot(p...,layout=(2,3),size=(1000,500))

u_noise = max.(u_ref .+ randn(size(u_ref)) .* σ_noise, 0.0)
#u_noise = u_ref .* (1.0 .+ randn(size(u_ref)) .* σ_noise)
p = []
for i = 1:size(u_noise,2)
    f = heatmap(xunique,yunique,reshape(u_noise[:,i],(nx,ny))',title="Frame$(i)",clims=(0,2))
    plot!(f,xlims=(0,Lx),ylims=(0,Ly))
    push!(p,f)
end
plot(p...,layout=(2,3),size=(1000,500))

## define forward model
ic = Vector(u_noise[:,1]) 
function eval_ad_model(df::DataFrame)
    n = nrow(df)        # number of particles
    pnames = names(df, x -> x[1] == 'ξ')
    evals = [spzeros(nx*ny, n_out+1) for _ in 1:n]

    for i in 1:n
        # reconstruct spatially-varying field(s) from this row's KL/SPDE coefficients
        ξ = collect(df[i, pnames])
        d_curr = sample_d(ξ)

        # forward solve
        u_curr = forward_solve_imex(d_curr, ic, vx, vy, adp, dt, nsteps, n_out)

        if minimum(u_curr) < -0.1 || 2.2 < maximum(u_curr)
            u_curr = forward_solve_imex(d_curr, ic, vx, vy, adp, dt/10, nsteps*10, n_out)
        end

        if minimum(u_curr) < -0.1 || 2.2 < maximum(u_curr) 
            @warn("Solution unstable!")
            print("min,max",minimum(u_curr),",",maximum(u_curr))
            u_curr .= NaN
        end

        evals[i] = u_curr
    end

    return evals
end
ad_model = Model(eval_ad_model, :AdvDiff)

# define rest for sTMCMC
function loglikelihood(df)
    evals = df.AdvDiff
    log_ll = zeros(length(evals))
    for n in eachindex(evals)
        if any(isnan.(evals[n]))
            log_ll[n] = NaN
        else
            log_ll[n] = -0.5 * sum(((u_noise .- evals[n]) ./ σ_noise) .^ 2)
        end
    end
    log_ll[isnan.(log_ll)] .= minimum(log_ll[.!isnan.(log_ll)]) / 10
    return log_ll
    #return [-0.5 * sum(((u_noise .- evals[n]) ./ σ_noise) .^ 2) for n in eachindex(evals)]
end

prior = RandomVariable.(Normal(), [Symbol("ξ$(i)") for i in 1:T_reco])

nrv_vec = [5:5:T_reco;]
particle_factor = 10
burnin = 5
seqtmcmc = SequentialTransitionalMarkovChainMonteCarlo(prior, nrv_vec, particle_factor, burnin)

# eval SeqTMCMC (model_calls = 79650)
stmcmc_samples, stmcmc_evidence = bayesianupdating(loglikelihood, [ad_model], seqtmcmc)
final_coefs_seq = Matrix(stmcmc_samples[:,names(prior)]) 
s_reco_seq = vec(sum(final_coefs_seq, dims=1)./size(final_coefs_seq,1))
d_reco_seq = sample_d(s_reco_seq)
u_reco_seq = forward_solve_imex(d_reco_seq, u0, vx, vy, adp, dt, nsteps, n_out)

# eval TMCMC (model_calls = 45300)
ntmcmc = nrv_vec[end] * particle_factor
tmcmc = TransitionalMarkovChainMonteCarlo(prior, ntmcmc, burnin)
tmcmc_samples, tmcmc_evidence = bayesianupdating(loglikelihood, [ad_model], tmcmc)
final_coefs = Matrix(tmcmc_samples[:,names(prior)]) 
s_reco = vec(sum(final_coefs, dims=1)./size(final_coefs,1))
d_reco = sample_d(s_reco)
u_reco = forward_solve_imex(d_reco, u0, vx, vy, adp, dt, nsteps, n_out)

# compute low order reference
s_ref_low = (grf.data.eigenfunc[:, 1:T_reco] * Diagonal(grf.data.eigenval[1:T_reco])) \ log.(d_ref ./ 0.1)
d_ref_low = sample_d(s_ref_low)

# Boxplot RF coefficients
final_coefs_seq .-= s_ref_low[1:T_reco]' 
final_coefs .-= s_ref_low[1:T_reco]' 
bp = StatsPlots.boxplot(repeat(1:T_reco, inner=size(final_coefs_seq,1)), vec(final_coefs_seq);
        xlabel="mode index", ylabel="ξ_reco - ξ_ref", outliers=false, label="seqTMCMC",linewidth=0)
StatsPlots.boxplot!(bp,repeat(1:T_reco, inner=size(final_coefs,1)), vec(final_coefs);
        outliers=false, label="TMCMC",fillalpha=0.75,linewidth=0,legend=:outertop,legend_columns=2)
plot!(bp,size=(800,500),guidefontsize=16,tickfontsize=14,legendfontsize=16,xticks=nrv_vec)


# Mixed Root Mean Square (MRMS) error
mrms(ref,approx) = sqrt(sum( ((ref .- approx)/(1.0 .+ abs.(ref))).^2 )/prod(size(ref))) 
d_mrms_seq = mrms(d_reco_seq,d_ref_low) 
d_mrms = mrms(d_reco,d_ref_low)
u_true_mrms_seq = mrms(Matrix(u_reco_seq),Matrix(u_ref)) 
u_true_mrms = mrms(Matrix(u_reco),Matrix(u_ref))
u_mrms_seq = mrms(Matrix(u_reco_seq),Matrix(u_noise)) 
u_mrms = mrms(Matrix(u_reco),Matrix(u_noise))
ulocal_true_mrms_seq = mrms.(Matrix(u_reco_seq),Matrix(u_ref)) 
ulocal_true_mrms = mrms.(Matrix(u_reco),Matrix(u_ref))
ulocal_mrms_seq = mrms.(Matrix(u_reco_seq),Matrix(u_noise)) 
ulocal_mrms = mrms.(Matrix(u_reco),Matrix(u_noise))
maxMRMS = max(maximum(ulocal_true_mrms_seq),maximum(ulocal_true_mrms))

# Heatmap diffusion fields
href  = heatmap(xunique,yunique,reshape(d_ref,(nx,ny))',clims=(0.0,1.5),title="Reference TO=$(T_ref)");
hreflow  = heatmap(xunique,yunique,reshape(d_ref_low,(nx,ny))',clims=(0.0,1.5),title="Reference T=$(T_reco)");
hreco_seq = heatmap(xunique,yunique,reshape(d_reco_seq,(nx,ny))',clims=(0.0,1.5),title="SeqTMCMC T=$(T_reco) \n MRMS_T=$(round(d_mrms_seq*100,digits=3))%");
hreco = heatmap(xunique,yunique,reshape(d_reco,(nx,ny))',clims=(0.0,1.5),title="TMCMC T=$(T_reco) \n MRMS_T=$(round(d_mrms*100,digits=3))%");
plot!(href,hreflow,hreco_seq,hreco,layout=(2,2),size=(800,800))

# Gif concentration field
anim = @animate for frame in axes(u_ref,2)
    uref  = heatmap(xunique,yunique,reshape(u_ref[:,frame],(nx,ny))',clims=(0.0,2.0),title="True u_ref TO=$(T_ref)");
    unoise  = heatmap(xunique,yunique,reshape(u_noise[:,frame],(nx,ny))',clims=(0.0,2.0),title="Noisy u_noise TO=$(T_ref)");
    ureco_seq = heatmap(xunique,yunique,reshape(u_reco_seq[:,frame],(nx,ny))',clims=(0.0,2.0),title="SeqTMCMC u T=$(T_reco) frame=$(frame)");
    ureco = heatmap(xunique,yunique,reshape(u_reco[:,frame],(nx,ny))',clims=(0.0,2.0),title="TMCMC T=$(T_reco) frame=$(frame)");
    ereco_seq = heatmap(xunique,yunique,reshape(ulocal_true_mrms_seq[:,frame],(nx,ny))',clims=(0.0,maxMRMS),title="SeqTMCMC MRMS(u,u_ref)=$(round(u_true_mrms_seq*100,digits=3))%");
    ereco = heatmap(xunique,yunique,reshape(ulocal_true_mrms[:,frame],(nx,ny))',clims=(0.0,maxMRMS),title="TMCMC MRMS(u,u_ref)=$(round(u_true_mrms*100,digits=3))%");
    plot!(uref,ureco_seq,ereco_seq,unoise,ureco,ereco,layout=(2,3),size=(1200,800))
end
gif(anim, fps=2)

