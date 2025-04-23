using Distributions
using Plots

alpha, beta = 0.5, 0.5
dist = Beta(alpha, beta)

p = plot(0:0.01:1, x -> pdf(dist, x), label="Beta($alpha, $beta)", legend=:topright)

alpha, beta = 0.1, 0.5
dist = Beta(alpha, beta)
plot!(0:0.01:1, x -> pdf(dist, x), label="Beta($alpha, $beta)", legend=:topright)

display(p)