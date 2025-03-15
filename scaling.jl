using LinearAlgebra, Statistics
using QuantumInformation
using FLoops
using DelimitedFiles
using Distributions
# BLAS.set_num_threads(1)

function rpeCritical!(A, L1::Integer, L2::Integer, a::Real, c::Real)
    d1, d2 = 2^L1, 2^L2
    μ = c * d1^a

    # if β == 1
    #     A = Matrix{Float64}(undef, (d1, d2))
    #     A[:,1] = rand(Normal(), d1)
    #     dist = Normal(0., sqrt(1/(1 + μ)))
    #     A[:, 2:d2] = rand(dist, (d1, d2-1))
    # elseif β == 2
    # A = Matrix{ComplexF64}(undef, (d1, d2))
    A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
    dist = Normal(0., sqrt(1/(1 + μ)))
    A[:, 2:d2] = rand(dist, (d1, d2-1)) + im*rand(dist, (d1, d2-1))
    # end

    # return A
end

function powerLawCritical!(A, L1::Integer, L2::Integer, a::Real, c::Real)

    d1, d2 = 2^L1, 2^L2
    μ = c * d1^a
    
    # A = Matrix{ComplexF64}(undef, (d1, d2))
    A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
    for j=2:d2
        for i=1:d1
            dist = Normal(0., sqrt(1/(1 + (i*(j-1)/μ))))
            A[i, j] = rand(dist) + im*rand(dist)
        end 
    end

end

function expDecayCritical!(A, L1::Integer, L2::Integer, a::Real, c::Real)
    d1, d2 = 2^L1, 2^L2
    μ = c * d1^a

    # A = Matrix{ComplexF64}(undef, (d1, d2))
    A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
    for j=2:d2
        for i=1:d1
            dist = Normal(0., sqrt(exp(-((j-1)*i/μ))))
            A[i, j] = rand(dist) + im*rand(dist)
        end 
    end

end

function entDynamics(a, c, α::Real, L1::Integer, L2::Integer)

    sampleSize = 10
    d1, d2 = 2^L1, 2^L2
    ent = Array{Float64}(undef, sampleSize)
    C = Matrix{ComplexF64}(undef, (d1, d2))
    ρ = similar(C)
    for k in 1:sampleSize
        # rpeCritical!(C, L1, L2, a, c)
        powerLawCritical!(C, L1, L2, a, c)
        # expDecayCritical!(C, L1, L2, a, c)
        mul!(ρ, C, C')
        ρ = ρ / tr(ρ)
        if α == 1
            ent[k] = vonneumann_entropy(ρ)/log(2)
        else
            ent[k] = renyi_entropy(ρ, α)/log(2)
        end
    end

    return mean(ent)

end

function comParamCrit(a, c, L1)
    d1 = 2^L1
    μ = c * d1^a
    
    l = -( (d1^2 - d1) / 0.5 ) * log(1 - (0.5 / (1 + μ)))
    d = exp(L1/2)
    t = log(L1 - 1)

    return l*t/d

end

function main()

    Larr = collect(20:2:24)
    alphaArr = LinRange(0.2,1.2,15)
    # measBEcrit = Vector{Float64}(undef, length(Larr))
    # measBEcrit = Matrix{Float64}(undef, length(Larr), 15)
    # measPEcrit = Matrix{Float64}(undef, length(Larr), length(alphaArr))
    measEEcrit = Matrix{Float64}(undef, length(alphaArr), length(Larr))
    # YBEcrit = Vector{Float64}(undef, length(Larr))

    # c = 0.1; a = 1.26
    c = 0.2; ens = "PE"

    for (j, a) in enumerate(alphaArr)
        for (i, l) in enumerate(Larr)
            lb = div(l, 2)
            # measBEcrit[i, j] = entDynamics(a, c, 1, lb, lb)
            # measPEcrit[i, j] = entDynamics(a, c, 1, lb, lb)
            measEEcrit[j, i] = entDynamics(a, c, 1, lb, lb)
            # YBEcrit[i] = comParamCrit(a, c, lb)
            print("l ==> $l ")
            flush(stdout)
        end
        print("a ==> $a\n")
        flush(stdout)
    end

    # writedlm("scalingCriticalBE,balanced,c=$c,a_medium.txt", [measEEcrit alphaArr])
    writedlm("r1critical,c=$c,$ens.txt", [alphaArr measure])
end

main()
