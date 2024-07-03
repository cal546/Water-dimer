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

lines = readlines(stdin);
path = split(lines[1],'#')[1]
n_add_epochs = parse(Int64,split(lines[2],'#')[1])


F_0 = load_object(string(path,"/Fourier.jld2"));              # Fourier model
loss_0 = load_object(string(path,"/loss.jld2"));              # training loss 
acceptance_0 = load_object(string(path,"/acceptance.jld2"));  # acceptance rate
data_train = load_object(string(path,"/train.jld2"));       # training data model model used
data_test = load_object(string(path,"/test.jld2"));         # testing data from same distribution of testing set

# parameters
A = [];
open(string(path,"/Parameters.txt")) do f
    lines = readlines(f)
    for l in lines
        push!(A,l)
    end
end

K = parse(Int64,split(A[2],'#')[1]); # read in number of nodes
λ = parse(Float64,split(A[5],'#')[1]); # read in regularization parameter
δ = parse(Float64,split(A[8],'#')[1]); # read in rwm
n_ω_steps = parse(Int64,split(A[9],'#')[1]); # read in number of steps between full β updates
ω_max = parse(Float64,split(A[10],'#')[1]); # read in ω max
adapt_covariance = parse(Bool,split(A[11],'#')[1]); # read in adaptive covariance


d = 2; # dimension

Σ0 = diagm(ones(d)); # initialization
n_burn = Int(.1 * n_add_epochs);
γ = optimal_γ(d);


β_solver! = (β, S, y, ω)-> ARFF.solve_normal_svd!(β, S, y; λ);

# sobolev weighted solver
function reg_β_solver!(β, S, y, λ, ω, r)
    N = length(y)
    β .= (S' * S + λ * N * diagm((norm.(ω)).^(2*r))) \ (S' * y)

end

r = 1;

opts = ARFFOptions(n_add_epochs, n_ω_steps, δ, n_burn, γ, ω_max,adapt_covariance, β_solver!, ARFF.mse_loss);

# Network training
Random.seed!(1000);


F = deepcopy(F_0);

Σ_mean, acceptance_rate, loss= train_rwm!(F, data_train,
    # batchsize,
    Σ0, opts,show_progress=true);

save_object(string(path,"/Fourier.jld2"), F);
save_object(string(path,"/loss.jld2"),vcat(loss_0,loss));
save_object(string(path,"/acceptance.jld2"),vcat(acceptance_0,acceptance_rate));

# make changes to Paramters.txt
open(string(path,"/Parameters.txt"),"w") do f
    println(f,A[1]);
    println(f,A[2]);
    println(f,A[3]);
    println(f,A[4]);
    println(f,A[5]);
    println(f,opts.loss(F, data_train.x, data_train.y),"        # training loss");
    println(f,opts.loss(F, data_test.x, data_test.y),"      # testing loss");
    println(f,A[8]);
    println(f,A[9]);
    println(f,A[10]);
    println(f,A[11]);
end




