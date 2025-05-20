# Computes the closest Gaussian approximation for a mixture of Gaussian
#
# 2025 by Ralf Herbrich
# Hasso-Plattner Institute

using LaTeXStrings
using Plots
using Distributions

struct MixtureOfGaussian
    weights::Vector{Float64}
    gaussians::Vector{Normal}
end

# computes the effective range of values of a mixture of Gaussian
function get_range(mog::MixtureOfGaussian)
    min_x = minimum(map(x -> x.μ, mog.gaussians) - 6 * map(x -> x.σ, mog.gaussians))
    max_x = maximum(map(x -> x.μ, mog.gaussians) + 6 * map(x -> x.σ, mog.gaussians))
    return min_x, max_x
end

# returns the PDF of a mixture of Gaussian
function mog_pdf(mog::MixtureOfGaussian, x::Float64)
    y = 0.0
    for i in 1:length(mog.weights)
        y += mog.weights[i] * pdf(mog.gaussians[i], x)
    end
    return y
end

# plots a mixture of Gaussian
function plot_mog(mog::MixtureOfGaussian; approx = nothing, title=nothing, ylim=nothing)
    min_x, max_x = get_range(mog)
    xs = range(min_x, max_x, length=1000)

    p = plot(xs,
        x -> mog_pdf(mog, x),
        label=false,
        xlabel=L"x",
        ylabel=L"p(x)",
        linewidth=3,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
        legendfontsize=16,
    )
    if !isnothing(approx)
        plot!(xs, 
            x -> pdf(approx, x), 
            linewidth=3,
            label = false,
            color=:red,
        )
    end
    if !isnothing(title)
        title!(title)
    end
    if !isnothing(ylim)
        ylims!(ylim)
    end

    display(p)
end

# computes the KL divergence between two distributions with discretized PDFs
function KL(p::Vector{Float64}, q::Vector{Float64})
    return sum(p .* log.(p ./ q))
end

# computes the KL divergence between two distributions with discretized PDFs
function KL_reverse(p::Vector{Float64}, q::Vector{Float64})
    return sum(q .* log.(q ./ p))
end

# computes the α-divergence between two distributions with discretized PDFs
function α_divergence(p::Vector{Float64}, q::Vector{Float64}; α=0.5)
    if α == 1
        return KL(p, q)
    elseif α == 0
        return KL_reverse(p, q)
    else
        return (1 - sum(q .* (p ./ q).^α)) / (α * (1 - α))
    end
end

# computes the mean of the mixture of Gaussian
function mean_and_variance(mog::MixtureOfGaussian)
    min_x, max_x = get_range(mog)
    xs = range(2*min_x, 2*max_x, length=n)
    p_mog = [mog_pdf(mog, x) for x in xs]
    p_mog = p_mog ./ sum(p_mog)

    μ = sum(xs .* p_mog)
    σ2 = sum((xs .- μ).^2 .* p_mog)
    return μ, σ2
end

# computes the closest Gaussian approximation for a mixture of Gaussian
function closest_gaussian(mog::MixtureOfGaussian; n = 10000, distance = KL)
    min_x, max_x = get_range(mog)
    xs = range(2*min_x, 2*max_x, length=n)
    p_mog = [mog_pdf(mog, x) for x in xs]
    p_mog = p_mog ./ sum(p_mog)

    μ = sum(xs .* p_mog)
    σ = sqrt(sum((xs .- μ).^2 .* p_mog))

    μs = range(μ / 1.5, μ*1.5, length=100)
    σs = range(σ / 2, σ * 2, length=100)

    smallest_distance = Inf
    best_μ = 0.0
    best_σ = 1.0

    for μ in μs
        for σ in σs
            p_normal = [pdf(Normal(μ, σ), x) for x in xs]
            p_normal = p_normal ./ sum(p_normal)
            d = distance(p_mog, p_normal)
            if d < smallest_distance
                smallest_distance = d
                best_μ = μ
                best_σ = σ
            end
        end
    end

    println("Best: $best_μ, $(best_σ^2)")

    return Normal(best_μ, best_σ)
end

# generates an animation of optimal Gaussian approximations of the α-divergence over all α in`αs`
# and plots the mean and variance of the optimal Gaussian approximation vs the true mean and variance
function plot_α_anim(mog; αs=range(-1.5, 1.5, length=30), 
                     anim_filename="~/Downloads/mog_anim.mp4",
                     μ_match_filename="~/Downloads/mog_μ_match.png",
                     σ2_match_filename="~/Downloads/mog_σ2_match.png")
    mog_μ, mog_σ2 = mean_and_variance(mog)

    μs = Vector{Float64}(undef, length(αs))
    σ2s = Vector{Float64}(undef, length(αs))
    
    anim = @animate for (i, α) in enumerate(αs)
        best_approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=α))
        μs[i], σ2s[i] = best_approx.μ, best_approx.σ^2
        plot_mog(
            mog, 
            approx = best_approx, 
            title="α = $(round(α, digits=1))",
            ylim=(0, 0.32)
        )
    end
    mp4(anim, anim_filename, fps=10)
    
    # plots the μs
    p = plot(
            αs, 
            μs, 
            linewidth=3, 
            color = :blue,
            label = L"\mu_{\mathrm{approximation}}",
            xlabel=L"\alpha", 
            ylabel=L"\mu_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs, 
        mog_μ * ones(length(αs)), 
        linewidth=3,
        label = L"\mu_p",
        color = :red,
    )
    display(p)
    savefig(μ_match_filename)
    
    p = plot(
            αs, 
            σ2s, 
            linewidth=3, 
            color = :blue,
            label=L"\sigma^2_{\mathrm{approximation}}",
            xlabel=L"\alpha", 
            ylabel=L"\sigma^2_{\mathrm{approximation}}",
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            legendfontsize=16,
        )
    plot!(
        αs, 
        mog_σ2 * ones(length(αs)), 
        linewidth=3,
        label = L"\sigma^2_p",
        color = :red,
    )
    display(p)    
    savefig(σ2_match_filename)
end

# mog = MixtureOfGaussian([0.3, 0.7], [Normal(0, 0.5), Normal(6, 1.5)])
mog = MixtureOfGaussian([0.3, 0.4, 0.3], [Normal(0, 0.5), Normal(3, 1), Normal(6, 1.5)])

plot_α_anim(mog,
            αs = range(-1, 2, length=50),
            anim_filename="~/Downloads/mog_anim.mp4",
            μ_match_filename="~/Downloads/mog_μ_match.svg",
            σ2_match_filename="~/Downloads/mog_σ2_match.svg"
)

plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-100)))
savefig("~/Downloads/mog_α_-100.svg")
plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=-1)))
savefig("~/Downloads/mog_α_-1.svg")
plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=0)))
savefig("~/Downloads/mog_α_0.svg")
plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=1)))
savefig("~/Downloads/mog_α_1.svg")
plot_mog(mog, approx = closest_gaussian(mog, distance = (p,q) -> α_divergence(p, q, α=100)))
savefig("~/Downloads/mog_α_100.svg")




