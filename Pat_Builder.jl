using Distributions
using LinearAlgebra
using NPZ

N = parse(Int, ARGS[1]);
nr_pats = parse(Int, ARGS[2]);
L = parse(Float64, ARGS[3]);


mu = zeros(Float64, nr_pats);
sigma = Matrix{Float64}(I, nr_pats, nr_pats);
d = MvNormal(mu, sigma);
patterns = rand(d, N);

u = Uniform(0,L);
positions = rand(u, 2, N);

npzwrite("Patterns.N$(ARGS[1]).nrpats$(ARGS[2]).L$(ARGS[3]).npz", Dict("Patterns_u"=> patterns, "Positions"=> positions));

#---------------------------------------------------------------------------------------------------
# ROUGH
#---------------------------------------------------------------------------------------------------
"""
pos = rand(u, 2, 100);
scatter(pos[1,:], pos[2,:])

#u = Uniform(0,L);
positions = [rand(u,2) for _ in 1:100];
x = [i[1] for i in positions];
y = [i[2] for i in positions];
scatter(x, y)
"""