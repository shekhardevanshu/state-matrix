begin
    using QuantumInformation
    using LinearAlgebra
    using Statistics
    using Distributions
    using FLoops
    using DelimitedFiles
end

numParams = 20
# numParams = 30
                    
# Hilbert-Schmidt Ensemble
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

function wishartMeasures(α::Real, sampleSize::Integer, β, L)

    measureWishart = Array{Float64}(undef, sampleSize)
    N = 2^L
    @floop for i in 1:sampleSize
        ρ = rand(HilbertSchmidtStates{β, 1}(N))
        if α == 1
            measureWishart[i] = vonneumann_entropy(ρ)/log(2)
        else
            measureWishart[i] = renyi_entropy(ρ, α)/log(2)
        end
    end
    # return mean(measureWishart)
    writedlm("wishart_alpha=$α,beta=$β,L=$L.txt", mean(measureWishart))

end

function defParamsPLE!(param_a, param_b, L1, L2)
    param_a[:] = LinRange(0., ∛(2*2^L1), numParams).^3
    param_b[:] = LinRange(0., ∛(2*2^L2), numParams).^3
end


#   Rosenzwig-Porter Ensemble
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

x1, x2 = 0.0, 1e7
# x1, x2 = 0.0, 1e3
param_rpe = LinRange(x1^(1/10), x2^(1/10), 400).^10
# param_rpe = LinRange(x1, x2, numParams)

function comParamRPE(L1::Integer, L2::Integer, param_a::Any)
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    γ = 0.25
    Y = -(d1*(d2-1)/(2*M*γ)) .* log.(abs.(1 .- (2*γ ./ (1 .+ param_a)))) #.- (N1/2*M*γ)*log(abs(1 - 2*γ))
    # Y = -N1*(N2-1)/(2*M*γ) .* log.(1 .- (2*γ .* μ))
    return Y
end

function rpeCol(L1::Integer, L2::Integer, μ::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2

    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(1 + μ)))
        A[:, 2:d2] = rand(dist, (d1, d2-1))
    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(1 + μ)))
        A[:, 2:d2] = rand(dist, (d1, d2-1)) + im*rand(dist, (d1, d2-1))
    end

    return A
end

function rpeCol(L1::Integer, L2::Integer, μ::Real, ν::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2
    
    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(sqrt(1 + μ)*sqrt(1 + ν))))
        A[:, 2:d2] = rand(dist, (d1, d2-1)) 

    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) .+ im*rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(sqrt(1 + μ)*sqrt(1 + ν))))
        A[:, 2:d2] = rand(dist, (d1, d2-1)) .+ im*rand(dist, (d1, d2-1))
    end

    return A
end

function comParamRPE(L1::Integer, L2::Integer, param_a::Any, param_b::Any)
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    γ = 0.25
    Y = Matrix{Float64}(undef, (numParams, numParams))
    for (i, μ) in enumerate(param_a)
        for (j, ν) in enumerate(param_b)
            Y[i, j] = -(d1*(d2-1)/(2*M*γ)) * log(abs(1 - (2*γ / (sqrt(1 + μ)*sqrt(1 + ν))))) #.- (N1/2*M*γ)*log(abs(1 - 2*γ))
        end
    end
    # Y = -N1*(N2-1)/(2*M*γ) .* log.(1 .- (2*γ .* μ))
    return Y
end

#   Power-Law Ensemble
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

ple_1 = 1200
# param_ple = LinRange(0., √1200, numParams).^2;
# param_ple = LinRange(0., (1200)^(1/5), numParams).^5
# param_ple = LinRange(0., ∛1200, numParams).^3
param_ple = LinRange(0., (ple_1)^(1/3), numParams).^3
# param_ple = LinRange(0.0001, ∛2400, numParams).^3
# param_ple1 = [10^y for y in range(log10(0.001), log10(2400), 20)]
# param_ple2 = [10^y for y in range(log10(2000), log10(10000), 5)]
# param_ple = LinRange(0., ∛10, numParams).^3
# param_ple = LinRange(0., 1200, numParams);

function powerLawCol(L1::Integer, L2::Integer, b::Float64, β::Integer)
    d1, d2 = 2^L1, 2^L2
    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j in 2:d2
            for i in 1:d1
                dist = Normal(0, sqrt(1/(1 + ((j-1)*i/b))))
                A[i, j] = rand(dist) 
            end    
        end
    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j in 2:d2
            for i in 1:d1
                dist = Normal(0, sqrt(1/(1 + ((j-1)*i/b))))
                A[i, j] = rand(dist) + im*rand(dist)
            end
        end
    end

    return A
end

function comParamPLEcol(L1::Integer, L2::Integer, param_a::Any)
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    gamma = 0.25

    Y = Array{Float64}(undef, length(param_a))
    for (i,b) in enumerate(param_a)
        y = -(d1/(2*M*gamma)) * sum([log(abs(1 - (2*gamma / (1 + (r/b)^2)))) for r=1:d2-1])
        Y[i] = y
    end
    return Y
end

function powerLawCol(L1::Integer, L2::Integer, a::Real, b::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2
    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(1/(1 + ((j-1)/b)*(i/a))))
                A[i, j] = rand(dist)
            end 
        end

    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(1/(1 + ((j-1)/b)*(i/a))))
                A[i, j] = rand(dist) + im*rand(dist)
            end 
        end
    end

    return A
end

function comParamPLEcol(L1::Integer, L2::Integer, param_a::Any, param_b::Any)
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    gamma = 0.25

    # Y = Matrix{Float64}(undef, (length(param_a), length(param_b)))
    Y = []

    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            y = 0
            for r1 in 1:d1
                for r2 in 1:d2-1
                    yt = 1 - (2*gamma / (1 + (r1/a)*(r2/b)))
                    if yt != 0
                        y += log(abs(yt))
                    end
                end
            end
            # y = -(β/(2*M*gamma)) * sum([log(abs(1 - (2*gamma / (1 + (r1/a)*(r2/b))))) for r1 in 1:d1-1 for r2 in 1:d2])
	    push!(Y, -(1/(2*M*gamma)) * y)
        end
    end

    return Y
end

#   Exponential Decay Ensemble
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

exp_1 = 1200
# param_exp = LinRange(0., √1200, numParams).^2;
# param_exp = LinRange(0., ∛1200, numParams).^3;
param_exp = LinRange(0., (exp_1)^(1/3), numParams).^3;
# param_exp = LinRange(0., ∛10, numParams).^3;
# param_exp = LinRange(0., 1200, numParams);

function expDecayCol(L1::Integer, L2::Integer, b, β::Integer)
    d1, d2 = 2^L1, 2^L2

    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j in 2:d2
            for i in 1:d1
                dist = Normal(0., sqrt(exp(-((j-1)*i/b))))
                A[i, j] = rand(dist)        
            end
        end
    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j in 2:d2
            for i in 1:d1
                dist = Normal(0., sqrt(exp(-((j-1)*i/b))))
                A[i, j] = rand(dist) + im*rand(dist)
            end
        end
    end

    return A
end

function comParamExpCol(L1::Integer, L2::Integer, param_a::Any)
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    gamma = 0.25

    Y = Array{Float64}(undef, length(param_a))
    for (i,b) in enumerate(param_a)
        y = -(d1/(2*M*gamma)) * sum([log(1 - (2*gamma / exp((r/b)^2))) for r=1:d2-1])
        Y[i] = y
    end
    return Y
end

function expDecayCol(L1::Integer, L2::Integer, a::Real, b::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2
    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(exp(-((j-1)/b)*(i/a))))
                A[i, j] = rand(dist)
            end 
        end

    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(exp(-((j-1)/b)*(i/a))))
                A[i, j] = rand(dist) + im*rand(dist)
            end 
        end
    end
    return A
end

function comParamExpCol(L1::Integer, L2::Integer, param_a::Any, param_b::Any)
    
    d1, d2 = 2^L1, 2^L2
    M = d1*d2
    gamma = 0.25

    # Y = Matrix{Float64}(undef, (numParams, numParams))
    Y = []

    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            y = -(1/(2*M*gamma)) * sum([log(abs(1 - (2*gamma / exp((r1/a)*(r2/b))))) for r1 in 1:d1 for r2 in 1:d2-1])
	    push!(Y, y)
        end
    end
    return Y
end

function getY(ens, l)

    Y = []

    try
        Y = readdlm("comParam,$ens,L=$l.txt")
        # print("hi")
    catch
        if ens == "RPE"
            Y = comParamRPE(l, l, param_rpe)
            writedlm("comParam,$ens,L=$l.txt", Y)
        elseif ens == "PLE"
            Y = comParamPLEcol(l, l, param_ple, param_ple)
            writedlm("comParam,$ens,L=$l.txt", Y)
        elseif ens == "EXP"
            Y = comParamExpCol(l, l, param_exp, param_exp)
            writedlm("comParam,$ens,L=$l.txt", Y)
        end
    end

    # print(size(Y))

    return Y
    
end

function getY(ens, l1, l2)

    Y = []

    param_a = zeros(numParams)
    param_b = similar(param_a)
    
    try
        Y = readdlm("./scaling/comParam,$ens,L1=$l,L2=$l2.txt")
        # print("hi")
    catch
        if ens == "RPE"
            Y = comParamRPE(l1, l2, param_rpe)
            writedlm("comParam,$ens,L1=$l1,L2=$l2.txt", Y)
        elseif ens == "PLE"
            # defParamsPLE!(param_a, param_b, l1, l2)
            Y = comParamPLEcol(l1, l2, param_ple, param_ple)
            # Y = comParamPLEcol(l1, l2, param_a, param_b)
            writedlm("./scaling/comParam,$ens,L1=$l1,L2=$l2.txt", Y)
        elseif ens == "EXP"
            # defParamsPLE!(param_a, param_b, l1, l2)
            Y = comParamExpCol(l1, l2, param_ple, param_ple)
            writedlm("comParam,$ens,L1=$l1,L2=$l2.txt", Y)
        end
    end

    # print(size(Y))

    return Y
    
end
