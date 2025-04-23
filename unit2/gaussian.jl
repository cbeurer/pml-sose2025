# Plots for visualization of inference with twi different parameterization of Gaussian distributions
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

using Plots
using Distributions
using LaTeXStrings
using Random

struct Gaussian
    τ::Float64
    ρ::Float64

    # default constructor
    Gaussian(τ, ρ) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ)...)
end

# Initializes a standard Gaussian 
Gaussian() = Gaussian(0, 1)

# Initializes a Gaussian from mean and variance
function GaussianFromMeanVariance(μ, σ²)
    return Gaussian(μ / σ², 1 / σ²)
end

# returns the mean of a Gaussian distribution
function μ(g::Gaussian)
    return g.τ / g.ρ
end

# returns the variance of a Gaussian distribution
function σ²(g::Gaussian)
    return 1 / g.ρ
end

# multiplies two Gaussian distributions
function Base.:*(g1::Gaussian, g2::Gaussian)
    return Gaussian(g1.τ + g2.τ, g1.ρ + g2.ρ)
end

# generate a training sample
function generate_sample(n::Int64; μ=0.0, σ²=1.0, β²=1.0)
    m = rand(Normal(μ, sqrt(σ²)), 1)[1]
    println("True mean: $m")
    return rand(Normal(m, sqrt(β²)), n)
end

# plot the prior/posteriors of the Gaussian distribution in location-scale parameterization
function plot_Gaussian_inference_μσ²(sample::Vector{Float64}, prior::Gaussian; β²=1.0)
    posteriors = Vector{Gaussian}()
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)
        end
        posterior = prior * GaussianFromMeanVariance(x, β²)
        push!(posteriors, posterior)
        prior = posterior
    end

    μs = map(d -> μ(d), posteriors)
    σ²s = map(d -> σ²(d), posteriors)
    p = plot(
        μs,
        σ²s,
        legend=false,
        yscale=:log10,
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    scatter!(μs, σ²s)
    xlabel!(L"\mu")
    ylabel!(L"\sigma^2")
    display(p)
end

# plot the prior/posteriors of the Gaussian distribution in natural parameterization
function plot_Gaussian_inference_τρ(sample::Vector{Float64}, prior::Gaussian; β²=1.0)
    posteriors = Vector{Gaussian}()
    for (i, x) in enumerate(sample)
        if i == 1
            push!(posteriors, prior)
        end
        posterior = prior * GaussianFromMeanVariance(x, β²)
        push!(posteriors, posterior)
        prior = posterior
    end

    τs = map(d -> d.τ, posteriors)
    ρs = map(d -> d.ρ, posteriors)
    p = plot(
        τs,
        ρs,
        legend=false,
        linewidth=3,
        color=:blue,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )
    scatter!(τs, ρs)
    xlabel!(L"\tau")
    ylabel!(L"\rho")
    display(p)
end

# creates an animation for the Delta dirac distribution
function plot_dirac_animation()
    
    x = range(-3, 3, length=1000)
    anim = @animate for i in 1:50
        plot(
            legend=false,
            xlim=(-3, 3),
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
        )        
        plot!(x, pdf(Normal(0, 1/i), x), color=:blue, linewidth=3)
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

# creates an animation for the Gaussian uniform
function plot_uniform_animation()
    
    x = range(-3, 3, length=1000)
    anim = @animate for i in 1:50
        plot(
            legend=false,
            xlim=(-3, 3),
            ylim=(0, 1),
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
        )        
        plot!(x, pdf(Normal(0, i), x), color=:blue, linewidth=3)
        ylabel!(L"p(x)")
        xlabel!(L"x")
    end

    return anim
end

# set the random seed to 42d
Random.seed!(42)
data = generate_sample(20, β²=0.1)

plot_Gaussian_inference_μσ²(data, Gaussian(0, 1), β²=0.1)
savefig("~/Downloads/gaussian_inference_μσ².svg")
plot_Gaussian_inference_τρ(data, Gaussian(0, 1), β²=0.1)
savefig("~/Downloads/gaussian_inference_τρ.svg")

anim = plot_dirac_animation()
gif(anim, "~/Downloads/dirac.gif", fps=10)
anim = plot_uniform_animation()
gif(anim, "~/Downloads/uniform.gif", fps=10)