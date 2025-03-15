using QuantumInformation
using LinearAlgebra, Random
using FLoops
BLAS.set_num_threads(1)
# using GLM, DataFrames
using DelimitedFiles
# using KernelDensity
using Statistics
include("ensembles.jl")

# L1, L2, sampleSize = 2, 2, 500;
samples = 100000
l = 10
# samplesDist = 100000
# paramRange = round(N/10)*10
# n = 10
β = 2

# ens_c = Dict("RPE" => 509.06, "PLE" => 56.10, "EXP" => 27.72) # r1, beta = 1

# function checkS_3(ensemble::Function, param)

#     coeffs = Matrix{Float64}(undef, (2,numParams))
#     # rsq = Array{Float64}(undef, numParams)
#     for (j, μ) in enumerate(param)
#         S_3 = Array{Float64}(undef, sampleSize)
#         S_12 = similar(S_3)
#         @floop for i in 1:sampleSize
#             C = ensemble(L1, L2, μ)
#             ρ = C*C'
#             ρ = ρ / tr(ρ)
#             S_12[i] = tr(ρ^2)^2
#             S_3[i] = tr(ρ^3)
#         end
#         data = DataFrame(X = S_12, Y = S_3)
#         reg = lm(@formula(Y ~ X), data);
#         coeffs[1, j] = coef(reg)[1]
#         coeffs[2, j] = coef(reg)[2]
#         # rsq[j] = r2(reg)
#     end
    
#     return coeffs
# end

# function checkS_3(ensemble::Function, param_a, param_b)

#     coeff = Matrix{Float64}(undef, (length(param_a), length(param_b)))
#     for (i, μ) in enumerate(param_a)
#         for (j, ν) in enumerate(param_b)
#             S_3 = Array{Float64}(undef, sampleSize)
#             S_12 = similar(S_3)
#             @floop for i in 1:sampleSize
#                 C = ensemble(L1, L2, μ, ν)
#                 ρ = C*C'
#                 ρ = ρ / tr(ρ)
#                 S_12[i] = tr(ρ^2)^2
#                 S_3[i] = tr(ρ^3)
#             end
#             if L1 >= 10
#                 data = DataFrame(X = sqrt.(S_12), Y = sqrt.(S_3))
#                 reg = lm(@formula(Y ~ X), data);
#                 coeff[i, j] = (coef(reg)[2])
#             else
#                 data = DataFrame(X = S_12, Y = S_3)
#                 reg = lm(@formula(Y ~ X), data);
#                 coeff[i, j] = coef(reg)[2]
#             end

#             # rsq[j] = r2(reg)
#         end
#     end
    
#     return coeff
# end

# function measDist(ens, a::Float64)
#     e = abs.(readdlm("./eigVals/$ens/beta=$β/L=$l/eigvals,a=$a,$ens,l=$l.txt"))
#     _, m2 = size(e)

#     # measures = Array{Float64}(undef, (3, samplesDist))
#     measures = Array{Float64}(undef, (2, m2))

#     # @floop for i in 1:m2
#         # C = ensemble(L1, L2, μ, β)
#         # ρ = C*C'
#         # ρ = ρ / tr(ρ)
#     #     measures[1,i] = real(tr(ρ^2))
#     #     measures[2,i] = real(vonneumann_entropy(ρ)/log(2))
#     #     measures[3,i] = real(renyi_entropy(ρ, 2)/log(2))
#     # end
#     measures[1,:] = [sum( e[:, i].^2 ) for i in 1:m2]
#     measures[2,:] = [-sum( e[:, i] .* log2.(e[:, i]) ) for i in 1:m2]
#     S3 = mean([sum( e[:, i].^3 ) for i in 1:m2])

#     return measures, S3
# end

# function measDist(ens, a::Float64, b::Float64)

#     e = abs.(readdlm("./eigVals/$ens/beta=$β/L=$l/eigvals,a=$a,b=$b,$ens,l=$l.txt"))
#     _, m2 = size(e)

#     measures = Array{Float64}(undef, (2, m2))
#     # S3 = Array{Float64}(undef, m2)

#     # @floop for i in 1:samplesDist
#     #     C = ensemble(L1, L2, μ, ν, β)
#     #     ρ = C*C'
#     #     ρ = ρ / tr(ρ)
#     #     measures[1,i] = real(tr(ρ^2))
#     #     measures[2,i] = real(vonneumann_entropy(ρ)/log(2))
#     #     measures[3,i] = real(renyi_entropy(ρ, 2)/log(2))
#     # end

#     measures[1,:] = [sum( e[:, i].^2 ) for i in 1:m2]
#     measures[2,:] = [-sum( e[:, i] .* log2.(e[:, i]) ) for i in 1:m2]
#     S3 = mean([sum( e[:, i].^3 ) for i in 1:m2])

#     return measures, S3
#     # return S3
# end

# function getIcBc(ensemble::Function, param_a, param_b, x, y)
#     dist_x = Array{Float64}(undef, (length(param_a), length(param_b)))
#     dist_y = similar(dist_x)

#     for (i,μ) in enumerate(param_a)
#         for (j, ν) in enumerate(param_b)
#             meas = Array{Float64}(undef, samplesDist)
#             @floop for k in 1:samplesDist
#                 C = ensemble(L1, L2, μ, ν)
#                 ρ = C*C'
#                 ρ = ρ / tr(ρ)
#                 meas[k] = tr(ρ^2)
#             end
#             U = kde(meas)
#             ik = InterpKDE(U)
#             dist_x[i, j] = pdf(ik, x)
#             dist_y[i, j] = pdf(ik, y)

#         end
#     end
#     return dist_x, dist_y
# end

# function checkApprox(ensemble::Function, param, ens)
#     arr1 = similar(param)
#     arr2 = similar(arr1)

#     a1 = Array{Float64}(undef, sampleSize)
#     a2 = similar(a1)

#     for (i,p) in enumerate(param)
    
#         @floop for m in 1:sampleSize
#             C = ensemble(l, l, p, 1)
#             ρ = C*C'
#             ρ = ρ / tr(ρ)
#             # if α == 1
#             e = eigvals(ρ)
#             e = abs.(e)
#             a1[m] = sum([(k*(log2(k))^2) for k in e])
#             a2[m] = vonneumann_entropy(ρ)/log(2)

#         end
#         arr1[i] = mean(a1)
#         arr2[i] = mean(a2)
#     end
#     writedlm("arr1,$ens.txt", arr1)
#     writedlm("arr2,$ens.txt", arr2)
# end

# function purDistWisart(d, sampleSize)
#     measureWishart = Array{Float64}(undef, sampleSize)
#     # C = Matrix{Float64}(undef, d, d)
#     # ρ = similar(C)
#     @floop for i in 1:sampleSize
#         # rand!(Normal(), C)
#         # LinearAlgebra.mul!(ρ, C, C')
#         # ρ ./= tr(ρ)
#         ρ = rand(HilbertSchmidtStates{β,1}(d))
#         # if α == 1
#         #     measureWishart[i] = vonneumann_entropy(ρ)/log(2)
#         # else
#         #     measureWishart[i] = renyi_entropy(ρ, α)/log(2)
#         # end
#         # e = eigvals(ρ)
#         # measureWishart[i] = sum(e.^2)
#         measureWishart[i] = tr(ρ^2)
#     end
#     # return mean(measureWishart)
#     writedlm("purityDistWishart,beta=$β,l=$d.txt", measureWishart)
# end

function entDynamics(ensemble::Function, param_a::Any, L1::Integer, L2::Integer, beta)

    measure = Array{Float64}(undef, (2, samples))

    ent1 = Array{Float64}(undef, samples)
    ent2 = Array{Float64}(undef, samples)
    # ρ = zeros(ComplexF64, (2^L1, 2^L2))
    # eVals = zeros(2^L1)
    @floop for k in 1:samples
        C = ensemble(L1, L2, param_a, beta)
        # mul!(ρ, C, C')
        ρ = C*C'
        ρ ./= tr(ρ)
        eVals = real.(abs.(eigvals(ρ)))

        ent1[k] = -sum([e*log2(e) for e in eVals])
        ent2[k] = sum([e^2 for e in eVals])
        if k % 10000 == 0
            print("completed sample = $k\n")
            flush(stdout)
        end
    end

    measure[2, :] = ent1
    measure[1, :] = ent2

    return measure
end

function entDynamics(ensemble::Function, param_a::Any, param_b::Any, L1::Integer, L2::Integer, beta)

    measure = Matrix{Float64}(undef, (2, samples))

    ent1 = Array{Float64}(undef, samples)
    ent2 = Array{Float64}(undef, samples)
    @floop for k in 1:samples
        C = ensemble(L1, L2, param_a, param_b, beta)
        ρ = C*C'
        ρ ./= tr(ρ)

        eVals = real.(abs.(eigvals(ρ)))

        ent1[k] = -sum([e*log2(e) for e in eVals])
        ent2[k] = sum([e^2 for e in eVals])

    end

    measure[2, :] = ent1
    measure[1, :] = ent2
    return measure
end

# RPE

function calcRPE()

    # coeff = checkS_3(rpeCol, param_rpe, param_rpe)
    # writedlm("coeff_RPE_two_param_L=$L1.txt", coeff)
    # coeff = readdlm("coeff_RPE_two_param_L=$L1.txt")

    # meas = readdlm("RPE_L=$l,one_param_1e7vn,β=$β.txt")
    # meas2 = readdlm("RPE_L=$l,one_param_r2,β=$β.txt")

    # Y = comParamRPE(L1, L2, param_rpe)
    Y = readdlm("comParam,RPE,L=$l.txt")

    # I = argmin(abs.(1e-6 .- Y))
    # J = argmin(abs.(1e-4 .- Y))
    # K = argmin(abs.(1e-3 .- Y))
    # L = argmin(abs.(1e-2 .- Y))
    # M = argmin(abs.(1e-1 .- Y))
    # N = argmin(abs.(1.0 .- Y))
    # N = argmax(Y)

    I = argmin(abs.(1e-5 .- Y))
    J = argmin(abs.(1e-4 .- Y))
    K = argmin(abs.(1e-3 .- Y))
    L = argmin(abs.(1e-2 .- Y))
    M = argmin(abs.(1e-1 .- Y))
    N = argmin(abs.(1 .- Y))
    # N = argmax(Y)

    @show Y[I]
    @show Y[J]
    @show Y[K]
    @show Y[L]
    @show Y[M]
    @show Y[N]

    # K = argmax(Y)


    # y = round(Y[I], digits = 7)
    ens = "rpe"

    # meas = entDynamics(rpeCol, param_rpe[I], l, l, β)
    # @show param_rpe[I]
    # writedlm("measDist_large_RPE_L=$l,Y=1e-5,beta=$β.txt", meas)

    # print("completed Y = 1e-5 \n")
    # flush(stdout)
    
    # meas = entDynamics(rpeCol, param_rpe[J], l, l, β)
    # @show param_rpe[J]
    # writedlm("measDist_large_RPE_L=$l,Y=1e-4,beta=$β.txt", meas)

    # print("completed Y = 1e-4 \n")
    # flush(stdout)

    # meas = entDynamics(rpeCol, param_rpe[K], l, l, β)
    # @show param_rpe[K]
    # writedlm("measDist_large_RPE_L=$l,Y=1e-3,beta=$β.txt", meas)

    # print("completed Y = 1e-3 \n")
    # flush(stdout)

    # meas = entDynamics(rpeCol, param_rpe[L], l, l, β)
    # @show param_rpe[L]
    # writedlm("measDist_large_RPE_L=$l,Y=1e-2,beta=$β.txt", meas)

    # print("completed Y = 1e-2 \n")
    # flush(stdout)

    # meas = entDynamics(rpeCol, param_rpe[M], l, l, β)
    # @show param_rpe[M]
    # writedlm("measDist_large_RPE_L=$l,Y=1e-1,beta=$β.txt", meas)

    # print("completed Y = 1e-1 \n")
    # flush(stdout)

    # meas = entDynamics(rpeCol, param_rpe[N], l, l, β)
    # @show param_rpe[L]
    # writedlm("measDist_large_RPE_L=$l,Y=1e0,beta=$β.txt", meas)

    # print("completed Y = 1e0 \n")
    # flush(stdout)

end

#PLE

function calcPLE()

    Y = readdlm("comParam,PLE,L=$l.txt")

    I = argmin(abs.(1e-5 .- Y))
    J = argmin(abs.(1e-4 .- Y))
    K = argmin(abs.(1e-3 .- Y))
    L = argmin(abs.(1e-2 .- Y))
    M = argmin(abs.(1e-1 .- Y))
    N = argmin(abs.(1 .- Y))

    @show Y[I]
    @show Y[J]
    @show Y[K]
    @show Y[L]
    @show Y[M]
    @show Y[N]

    # meas = entDynamics(powerLawCol, param_ple[I[1]], param_ple[I[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e-5,beta=$β.txt", meas)

    # print("completed Y = 1e-5 \n")
    # flush(stdout)

    # @show param_ple[I[1]], param_ple[I[2]]
    
    # meas = entDynamics(powerLawCol, param_ple[J[1]], param_ple[J[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e-4,beta=$β.txt", meas)

    # print("completed Y = 1e-4 \n")
    # flush(stdout)

    # @show param_ple[J[1]], param_ple[J[2]]

    # meas = entDynamics(powerLawCol, param_ple[K[1]], param_ple[K[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e-3,beta=$β.txt", meas)

    # print("completed Y = 1e-3 \n")
    # flush(stdout)

    # @show param_ple[K[1]], param_ple[K[2]]

    # meas = entDynamics(powerLawCol, param_ple[L[1]], param_ple[L[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e-2,beta=$β.txt", meas)

    # print("completed Y = 1e-2 \n")
    # flush(stdout)

    # @show param_ple[L[1]], param_ple[L[2]]

    # meas = entDynamics(powerLawCol, param_ple[M[1]], param_ple[M[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e-1,beta=$β.txt", meas)

    # print("completed Y = 1e-1 \n")
    # flush(stdout)

    # @show param_ple[M[1]], param_ple[M[2]]

    # meas = entDynamics(powerLawCol, param_ple[N[1]], param_ple[N[2]], l, l, β)
    # writedlm("measDist_large_PLE_L=$l,Y=1e0,beta=$β.txt", meas)

    # print("completed Y = 1e0 \n")
    # flush(stdout)

    # @show param_ple[N[1]], param_ple[N[2]]

end

#EXP

function calcEXP()

    Y = readdlm("comParam,EXP,L=$l.txt")

    I = argmin(abs.(1e-5 .- Y))
    J = argmin(abs.(1e-4 .- Y))
    K = argmin(abs.(1e-3 .- Y))
    L = argmin(abs.(1e-2 .- Y))
    M = argmin(abs.(1e-1 .- Y))
    N = argmin(abs.(1 .- Y))

    @show Y[I]
    @show Y[J]
    @show Y[K]
    @show Y[L]
    @show Y[M]
    @show Y[N]

    # meas = entDynamics(expDecayCol, param_exp[I[1]], param_exp[I[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e-5,beta=$β.txt", meas)

    # @show param_exp[I[1]], param_exp[I[2]]
    
    # meas = entDynamics(expDecayCol, param_exp[J[1]], param_exp[J[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e-4,beta=$β.txt", meas)

    # @show param_exp[J[1]], param_exp[J[2]]

    # meas = entDynamics(expDecayCol, param_exp[K[1]], param_exp[K[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e-3,beta=$β.txt", meas)

    # @show param_exp[K[1]], param_exp[K[2]]

    # meas = entDynamics(expDecayCol, param_exp[L[1]], param_exp[L[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e-2,beta=$β.txt", meas)

    # @show param_exp[L[1]], param_exp[L[2]]

    # meas = entDynamics(expDecayCol, param_exp[M[1]], param_exp[M[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e-1,beta=$β.txt", meas)

    # @show param_exp[M[1]], param_exp[M[2]]

    # meas = entDynamics(expDecayCol, param_exp[N[1]], param_exp[N[2]], l, l, β)
    # writedlm("measDist_large_EXP_L=$l,Y=1e0,beta=$β.txt", meas)

    # @show param_exp[N[1]], param_exp[N[2]]
end

# calcEXP()
# calcRPE()
calcPLE()
