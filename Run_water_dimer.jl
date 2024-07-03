using LaTeXStrings
using Random
using Statistics
using Distributions
using Printf
using LinearAlgebra
using LaTeXStrings
using DelimitedFiles
using Flux
using ARFF
using JLD2

# Read input passed in 
A = [];
path = "";
lines = readlines(stdin);

for i in 1:length(lines)
    global path
    if (i == 1)
        println(lines[i]);
        path = strip(split(lines[i],'#')[1])
    else
        println(lines[i]);
        l = split(lines[i],'#')[1];
        if (l != "")
            push!(A,parse(Float64,l));
        end
    end
end

# Number of Network Nodes
K = Int(A[1]);
# choosing number of (exclusive) training and testing points
n_train = Int(A[2]);
n_test = Int(A[3]);
# number of epochs
n_epochs = Int(A[4]); 
# regularization; Small as possible without overfitting
λ = A[5];

# data input
data_ = readdlm("CCpol23_data_sd15_150000.txt")
@show size(data_);

# formatting data
n_total, _ = size(data_);
d = 6;
@show n_total;
X_ = [data_[i,1:d].*prepend!(ones(d-1),1) for i in 1:n_total]
y_ = data_[:,d+1];

#outlier cleanup --- in my opinion approx. 35 for upper is good and -10 for lower
upper_energy_threshold = 35;
lower_energy_threshold = -10;

#apply energy filter
retain_idx = findall( (y_ .< upper_energy_threshold .&& y_ .> lower_energy_threshold));
@show n_total = length(retain_idx);
X = deepcopy(X_[retain_idx]);
y = deepcopy(y_[retain_idx]);

# selects them from big data set
Random.seed!(100);
idx = randperm(n_total)[1:(n_train+n_test)];
idx_train = idx[1:n_train];
idx_test = idx[n_train+1:end];

# NEED DEEPCOPY TO AVOID ALIASING ISSUE
data_full = DataSet(deepcopy(X), deepcopy(y));
data_train = DataSet(deepcopy(X[idx_train]),deepcopy(y[idx_train]));
data_test = DataSet(deepcopy(X[idx_test]),deepcopy(y[idx_test]));

# !-- scales existing data set and saves over original data set.
scalings = get_scalings(data_full);
scale_data!(data_full, scalings);
scale_data!(data_train, scalings);
scale_data!(data_test, scalings);

# Initializes Network Parameters
Random.seed!(200)

#Initial values are important. !!!! Include in parameters.txt
S_ω = 0.3; # scale of ω values
F0 = FourierModel([randn(ComplexF64) for _ in 1:K], # β values
    [rand(Uniform(-S_ω,S_ω),d) for _ in 1:K]); # ω values


@show δ = 2.4^2/(15*d); # rwm step size (d is input dimension). I wouldn't play around with it (came in paper).
Σ0 = diagm(ones(d)); # initialization
n_ω_steps = 5; # number of steps between full β updates
n_burn = Int(.1 * n_epochs);
γ = optimal_γ(d);
# γ = 16;
@show γ;
ω_max = Inf; # haven't found a scenario where this needs to be changes. 
adapt_covariance = true; # paper says this works well.

β_solver! = (β, S, y, ω)-> ARFF.solve_normal_svd!(β, S, y; λ);

# sobolev weighted solver
function reg_β_solver!(β, S, y, λ, ω, r)
    N = length(y)
    β .= (S' * S + λ * N * diagm((norm.(ω)).^(2*r))) \ (S' * y)
end

r = 1;

# Minibatching size (if necessary). Network performs better on whole data set, but sometimes this is unavoidable due to time.
# @show batchsize = 9000

opts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max,adapt_covariance, β_solver!, ARFF.mse_loss);

# Network training
Random.seed!(1000);
F = deepcopy(F0);

Σ_mean, acceptance_rate, loss= train_rwm!(F, data_train,
    # batchsize,
    Σ0, opts,show_progress=true);
#
@show Σ_mean

# saving
mkpath(path);

save_object(string(path,"/Fourier.jld2"), F);
save_object(string(path,"/loss.jld2"),loss);
save_object(string(path,"/acceptance.jld2"),acceptance_rate);
save_object(string(path,"/train.jld2"),data_train);
save_object(string(path,"/test.jld2"),data_test);
save_object(string(path,"/scaling.jld2"),scalings)

open(string(path,"/Parameters.txt"),"w") do f
    println(f,path,"        # path it was saved under");
    println(f,K,"           # number of nodes");
    println(f,n_train,"             # number of training points");
    println(f,n_test,"          # number of testing points");
    println(f,λ,"           # regularization paramter");
    println(f,opts.loss(F, data_train.x, data_train.y),"        # training loss");
    println(f,opts.loss(F, data_test.x, data_test.y),"      # testing loss");
    println(f,δ,"       # rwm step size");
    println(f,n_ω_steps,"           # number of steps between full β updates");
    println(f,ω_max,"           # maximum value ω could take");
    println(f,adapt_covariance,"            # adaptive covariance");
    println(f,S_ω,"              # scale of initial ω values")
end
