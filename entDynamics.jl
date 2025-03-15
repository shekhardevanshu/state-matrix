begin
    # using QuantumInformation
    using LinearAlgebra
    BLAS.set_num_threads(1)
    using Statistics
    using DelimitedFiles
    using FLoops
    using Printf
    # using StatsPlots
    include("ensembles.jl")
end

sampleSize = 1000
p = Dict("vn" => 1, "r2" => 2) #renyi entropy
# p = Dict("S_2" => 2)

β = 2

function entDynamics(ensemble::Function, param_a::Any, α::Real, L1::Integer, L2::Integer, beta)

    measure = Array{Float64}(undef, length(param_a))

    for (j, μ) in enumerate(param_a)
        ent = Array{Float64}(undef, sampleSize)
        @floop for k in 1:sampleSize
            C = ensemble(L1, L2, μ, beta)
            ρ = C*C'
            ρ = ρ / tr(ρ)
            if α == 1
                ent[k] = vonneumann_entropy(ρ)/log(2)
            else
                ent[k] = renyi_entropy(ρ, α)/log(2)
            end
        end
        measure[j] = mean(ent)
    end

    return measure
end

function entDynamics(ensemble::Function, param_a::Any, param_b::Any, α::Real, L1::Integer, L2::Integer, beta)

    measure = Matrix{Float64}(undef, (length(param_a), length(param_b)))

    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            ent = Array{Float64}(undef, sampleSize)
            @floop for k in 1:sampleSize
                C = ensemble(L1, L2, a, b, beta)
                ρ = C*C'
                ρ = ρ / tr(ρ)
                if α == 1
                    ent[k] = vonneumann_entropy(ρ)/log(2)
                else
                    ent[k] = renyi_entropy(ρ, α)/log(2)
                end
            end
            measure[i, j] = mean(ent)
        end
    end
    return measure
end

function avgRhoMoments(ensemble::Function, param_a::Any, param_b::Any, α::Real, L1::Integer, L2::Integer)

    measure = Matrix{Float64}(undef, (length(param_a), length(param_b)))

    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            ent = Array{Float64}(undef, sampleSize)
            @floop for k in 1:sampleSize
                C = ensemble(L1, L2, a, b, β)
                ρ = C*C'
                ρ = ρ / tr(ρ)
                if α == 2
                    ent[k] = 1 / tr(ρ^α)
                else
                    ent[k] = tr(ρ^α)
                end
            end
            measure[i, j] = mean(ent)
        end
    end
    return measure
end

function avgRhoMomentsWihtS1(ensemble::Function, comParam::Function, param_a::Any, param_b::Any, α::Real, L1::Integer, L2::Integer)

    Y = []
    s1 = LinRange(0.1, 1, 50)
    measure = Matrix{Float64}(undef, (length(param_a)*length(param_b), length(s1)))

    i = 1
    for a in param_a
        for b in param_b
            for (j, t) in enumerate(s1)
                ent = Array{Float64}(undef, sampleSize)
                @floop for k in 1:sampleSize
                    C = ensemble(L1, L2, a, b, β)
                    ρ = C*C'
                    ρ = t * ρ / tr(ρ)
                    if α == 2
                        ent[k] = 1 / tr(ρ^α)
                    else
                        ent[k] = tr(ρ^α)
                    end
                end
            measure[i, j] = mean(ent)
            end
            # push!(measure, mean(ent))
            i += 1
            push!(Y, comParam(L1, L2, a, b))
        end
    end
    writedlm("avgInvS2vsY_S1.txt", measure)
    writedlm("comParamInvS2vsY_S1.txt", Y)
    # return measure
end

function avgRhoMoments(ensemble::Function, param_a::Any, α::Real, L1::Integer, L2::Integer)

    measure = Array{Float64}(undef, length(param_a))

    for (i,a) in enumerate(param_a)
        ent = Array{Float64}(undef, sampleSize)
        @floop for k in 1:sampleSize
            C = ensemble(L1, L2, a, β)
            ρ = C*C'
            ρ = ρ / tr(ρ)
            if α == 2
                ent[k] = 1 / tr(ρ^α)
            else
                ent[k] = tr(ρ^α)
            end
        end
        measure[i] = mean(ent)
    end
    return measure
end

function checkApproxForR1(ensemble::Function, param_a::Any, param_b::Any, L1::Integer, L2::Integer, ens)
    # measure = Matrix{Float64}(undef, (2, length(param_a)*length(param_b)))
    measure = []
    Y = []
    # N = 2^L1
    count = 0 
    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            T1 = Array{Float64}(undef, sampleSize)
            # T2 = similar(T1)
            # Δ = similar(T1)
            # R12 = similar(T2)
            @floop for k in 1:sampleSize
                C = ensemble(L1, L2, a, b, β)
                ρ = C*C'
                ρ = ρ / tr(ρ)
                λ = abs.(eigvals(ρ))
                T1[k] = -sum(log2.(λ))
                # T2[k] = sum(λ .* ((log2.(λ)).^2))
                # Δ[k] = sum([abs(λ[u] - λ[v]) for u in 1:N for v in u+1:N])
                # R12[k] = (vonneumann_entropy(ρ)/log(2))^2
            end
            # c = length(param_b)*(i-1) + j
            # measure[1, c] = mean(T2)
            # measure[2, c] = mean(T2)
            # measure[3, c] = mean(Δ)
            # measure[2, c] = mean(R12)
	    print("cont = $count \n")
	    flush(stdout)
	    count += 1
            push!(measure, mean(T1))
        end
    end

    if ens == "exp"
        Y = comParamExpCol(L1, L2, param_a, param_b)
    elseif ens == "ple"
        Y = comParamPLEcol(L1, L2, param_a, param_b)
    end
    writedlm("R0,$ens,L=$L1,beta=$β.txt", [measure vec(Y)])
    # return measure
end

function checkApproxForR1(ensemble::Function, param_a::Any, L1::Integer, L2::Integer, ens)
    # measure = Matrix{Float64}(undef, (2, length(param_a)))
    measure = []

    for (i,a) in enumerate(param_a)
        T1 = Array{Float64}(undef, sampleSize)
        # T2 = similar(T1)
        # Δ = similar(T1)
        # R12 = similar(T2)
        @floop for k in 1:sampleSize
            C = ensemble(L1, L2, a, β)
            ρ = C*C'
            ρ = ρ / tr(ρ)
            λ = abs.(eigvals(ρ))
            T1[k] = -sum(log2.(λ))
            # T2[k] = sum(λ .* ((log2.(λ)).^2))
            # Δ[k] = sum([abs(λ[u] - λ[v]) for u in 1:L1 for v in u+1:L1])
            # R12[k] = (vonneumann_entropy(ρ)/log(2))^2
        end
        # measure[1, i] = mean(T2)
        # measure[2, i] = mean(T2)
        # measure[3, i] = mean(Δ)
        # measure[2, i] = mean(R12)
        push!(measure, mean(T1))
    end
    Y = comParamRPE(L1, L2, param_a)
    # return measure
    # writedlm("approxR_1,$ens.txt", measure)
    writedlm("R0,$ens,L=$L1,beta=$β.txt", [measure Y])
end

function keepEigvals(ensemble::Function, param_a::Any, param_b::Any, L1::Integer, L2::Integer, ens::String)
    for a in param_a
        for b in param_b
            if isfile("./eigVals/$ens/eigvals,a=$a,b=$b,$ens,l=$L1.txt")
                print("file exists \n")
                flush(stdout)
            else	
                v = Matrix{Float64}(undef, (2^L1, sampleSize))

                @floop for k in 1:sampleSize
                    C = ensemble(L1, L2, a, b, β)
                    ρ = C*C'
                    ρ = ρ / tr(ρ)
                    λ = abs.(eigvals(ρ))
                    v[:, k] = λ
                end
                writedlm("./eigVals/$ens/eigvals,a=$a,b=$b,$ens,l=$L1.txt", v)
            end

        end
    end
end

function keepEigvals(ensemble::Function, param_a::Any, L1::Integer, L2::Integer, ens::String)
    for a in param_a

        astr = @sprintf("%.3e", a)
        
        if isfile("./eigVals/$ens/eigvals,a=$astr,$ens,l=$L1.txt")
            print("file exists \n")
            flush(stdout)
        else
            v = Matrix{Float64}(undef, (2^L1, sampleSize))

            @floop for k in 1:sampleSize
                C = ensemble(L1, L2, a, β)
                ρ = C*C'
                ρ = ρ / tr(ρ)
                λ = abs.(eigvals(ρ))
                v[:, k] = λ
            end

            writedlm("./eigVals/$ens/eigvals,a=$astr,$ens,l=$L1.txt", v)
        end
	
    end
end

function calcFlucs(param_a::Any, param_b::Any, l::Integer, ens::String)
    
    # delfluc_vn = []
    # fluc_vn = []
    # delfluc_r2 = []
    fluc_r2 = []
    # avg_r2 = []
    # avg_vn = []
    # fluc_s2 = []
    # avg_s2 = []
    # delfluc_s2 = []

    for b in param_b
        for a in param_a
            e = abs.(readdlm("./eigVals/$ens/beta=$β/L=$l/eigvals,a=$a,b=$b,$ens,l=$l.txt"))
            _, m2 = size(e)
            # vn = [-sum( e[:, i] .* log2.(e[:, i]) ) for i in 1:m2]
            # r2 = [-log2.( sum( e[:, i].^2 ) ) for i in 1:m2]
            s2 = [sum( e[:, i].^2 ) for i in 1:m2]
	        r2 = -log2.(s2)
            # push!(avg_r2, mean(r2))
            # push!(avg_vn, mean(vn))
            # push!(delfluc_vn, std(vn) / abs.(mean(vn)) )
            # push!(fluc_vn, std(vn))
            # push!( delfluc_r2, std(r2) / abs.(mean(r2)) )
            push!(fluc_r2, std(r2))
            # push!(fluc_s2, std(s2))
            # push!( delfluc_s2, std(s2) / abs.(mean(s2)) )
            # push!(avg_s2, mean(s2))
        end
    end
    # writedlm("vn_delfluc,$ens,l=$l,beta=$β.txt", delfluc_vn)
    # writedlm("vn_fluc,$ens,l=$l,beta=$β.txt", fluc_vn)
    # writedlm("vn_avg,$ens,l=$l,beta=$β.txt", avg_vn)
    # writedlm("r2_delfluc,$ens,l=$l,beta=$β.txt", delfluc_r2)
    writedlm("r2_fluc,$ens,l=$l,beta=$β.txt", fluc_r2)
    # writedlm("r2_avg,$ens,l=$l,beta=$β.txt", avg_r2)
    # writedlm("s2_delfluc,$ens,l=$l,beta=$β.txt", delfluc_s2)
    # writedlm("s2_fluc,$ens,l=$l,beta=$β.txt", fluc_s2)
    # writedlm("s2_avg,$ens,l=$l,beta=$β.txt", avg_s2)
    
end

function calcFlucs(param_a::Any, l::Integer, ens::String)
    
    # delfluc_vn = []
    # fluc_vn = []
    # delfluc_r2 = []
    fluc_r2 = []
    # avg_r2 = []
    # avg_vn = []
    # fluc_s2 = []
    # avg_s2 = []
    # delfluc_s2 = []

    for a in param_a
        e = abs.(readdlm("./eigVals/$ens/beta=$β/L=$l/eigvals,a=$a,$ens,l=$l.txt"))
        _, m2 = size(e)
        # vn = [-sum( e[:, i] .* log2.(e[:, i]) ) for i in 1:m2]
        # r2 = [-log2.( sum( e[:, i].^2 ) ) for i in 1:m2]
        s2 = [sum( e[:, i].^2 ) for i in 1:m2]
        r2 = -log2.(s2)
        # push!(avg_r2, mean(r2))
        # push!(avg_vn, mean(vn))
        # push!( delfluc_vn, std(vn) / abs.(mean(vn)) )
        # push!(fluc_vn, std(vn))
        # push!( delfluc_r2, std(r2) / abs.(mean(r2)) )
        push!(fluc_r2, std(r2))
        # push!(fluc_s2, std(s2))
        # push!( delfluc_s2, std(s2) / abs.(mean(s2)) )
        # push!(avg_s2, mean(s2))
    end
    # writedlm("vn_delfluc,$ens,l=$l,beta=$β.txt", delfluc_vn)
    # writedlm("vn_fluc,$ens,l=$l,beta=$β.txt", fluc_vn)
    # writedlm("vn_avg,$ens,l=$l,beta=$β.txt", avg_vn)
    # writedlm("r2_delfluc,$ens,l=$l,beta=$β.txt", delfluc_r2)
    writedlm("r2_fluc,$ens,l=$l,beta=$β.txt", fluc_r2)
    # writedlm("r2_avg,$ens,l=$l,beta=$β.txt", avg_r2)
    # writedlm("s2_delfluc,$ens,l=$l,beta=$β.txt", delfluc_s2)
    # writedlm("s2_fluc,$ens,l=$l,beta=$β.txt", fluc_s2)
    # writedlm("s2_avg,$ens,l=$l,beta=$β.txt", avg_s2)
    
end

function scalingLawPLE(a, b, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        C = powerLawCol(L1, L2, a, b, β)
        # C = expDecayCol(L1, L2, a, b, β)
        ρ = C*C'
        ρ = ρ / tr(ρ)
        if α == 1
            ent[k] = vonneumann_entropy(ρ)/log(2)
        else
            ent[k] = renyi_entropy(ρ, 2)/log(2)
        end
    end
    return mean(ent)
    # return std(ent)

end

function scalingLawEXP(a, b, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        # C = powerLawCol(L1, L2, a, b, β)
        C = expDecayCol(L1, L2, a, b, β)
        ρ = C*C'
        ρ = ρ / tr(ρ)
        if α == 1
            ent[k] = vonneumann_entropy(ρ)/log(2)
        else
            ent[k] = renyi_entropy(ρ, 2)/log(2)
        end
    end

    return mean(ent)
    # return std(ent)

end

function scalingLawRPE(a, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        C = rpeCol(L1, L2, a, β)
        ρ = C*C'
        ρ = ρ / tr(ρ)
        if α == 1
            ent[k] = vonneumann_entropy(ρ)/log(2)
        else
            ent[k] = renyi_entropy(ρ, 2)/log(2)
        end
    end
    return mean(ent)
    # return std(ent)

end

function participationEntropyPLE(a, b, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        C = powerLawCol(L1, L2, a, b, β)
        # C = expDecayCol(L1, L2, a, b, β)
        C ./= norm(C, 2)
        Pr = vec(C .* conj(C))
        if α == 1
            ent[k] = sum([-p*log2(p) for p in Pr])
        else
            ent[k] = -log2(sum(Pr .^ 2))
        end
    end

    return mean(ent)
end

function participationEntropyEXP(a, b, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        C = expDecayCol(L1, L2, a, b, β)
        C ./= norm(C, 2)
        Pr = vec(C .* conj(C))
        if α == 1
            ent[k] = sum([-p*log2(p) for p in Pr])
        else
            ent[k] = -log2(sum(Pr .^ 2))
        end
    end

    return mean(ent)
end

function participationEntropyRPE(a, L1, L2, α)

    ent = Array{Float64}(undef, sampleSize)

    @floop for k in 1:sampleSize
        C = rpeCol(L1, L2, a, β)

        C ./= norm(C, 2)
        Pr = vec(C .* conj(C))
        if α == 1
            ent[k] = sum([-p*log2(p) for p in Pr])
        else
            ent[k] = -log2(sum(Pr .^ 2))
        end
    end

    return mean(ent)
end

function participationRatio(ensemble::Function, param_a::Any, L1::Integer, L2::Integer, beta, e)

    measure = Array{Float64}(undef, length(param_a))

    for (j, μ) in enumerate(param_a)
        ent = Array{Float64}(undef, sampleSize)
        @floop for k in 1:sampleSize
            C = ensemble(L1, L2, μ, beta)
            C ./= norm(C, 2)
            Pr = vec(C .* conj(C))
            ent[k] = sum(Pr .^ 2)
        end
        measure[j] = mean(ent)
    end
    writedlm("participationRatio$e,L=$L1,beta=$β.txt", measure)
    # return measure
end

function participationRatio(ensemble::Function, param_a::Any, param_b::Any, L1::Integer, L2::Integer, beta, e)

    measure = Matrix{Float64}(undef, (length(param_a), length(param_b)))

    for (i,a) in enumerate(param_a)
        for (j,b) in enumerate(param_b)
            ent = Array{Float64}(undef, sampleSize)
            @floop for k in 1:sampleSize
                C = ensemble(L1, L2, a, b, beta)
                C ./= norm(C, 2)
                Pr = vec(C .* conj(C))
                ent[k] = sum(Pr .^ 2)
            end
            measure[i, j] = mean(ent)
        end
    end
    writedlm("participationRatio$e,L=$L1,beta=$β.txt", measure)
    # return measure
end

# L = 20

# for l in 1:10
    
#     for (i, j) in p
#         # meas_1 = entDynamics(expDecayCol, param_exp, param_exp, j, l, l);
#         # meas_1 = entDynamics(expDecayCol, param_exp, param_exp, j, l, L-l, β);
#         # keepEigvals(expDecayCol, param_exp, param_exp, l, L-l, "exp")
#         participationRatio(expDecayCol, param_exp, param_exp, l, L-l, β, "exp")
#         # meas_1 = entDynamics(expDecayCol, param_exp, j, l, l);
#         # writedlm("EXP_L1=$l,L2=$(L-l),$i,beta=$β.txt", meas_1)
#         # meas_2 = entDynamics(powerLawCol, param_ple, param_ple, j, l, l);
#         # meas_2 = entDynamics(powerLawCol, param_ple, param_ple, j, l, L-l, β);
#         # keepEigvals(powerLawCol, param_ple, param_ple, l, L-l, "ple")
#         participationRatio(powerLawCol, param_ple, param_ple, l, L-l, β, "ple")
#         # meas_2 = entDynamics(powerLawCol, param_ple, j, l, l);
#         # writedlm("./data_final/PLE_L=$l,two_param_$i,beta=$β.txt", meas_2)
#         # writedlm("PLE_L1=$l,L2=$(L-l),$i,beta=$β.txt", meas_2)
#         # meas_3 = entDynamics(rpeCol, param_rpe, param_rpe, j, l, l);
#         # meas_3 = entDynamics(rpeCol, param_rpe, j, l, l);
#         # meas_3 = entDynamics(rpeCol, param_rpe, j, l, L-l, β);
#         # keepEigvals(rpeCol, param_rpe, l, L-l, "rpe")
#         participationRatio(rpeCol, param_rpe, l, L-l, β, "rpe")
#         # writedlm("./data_final/RPE_L=$l,one_param_$i,beta=$β.txt", meas_3)
#         # writedlm("RPE_L1=$l,L2=$(L-l),$i,beta=$β.txt", meas_3)
#     end

# end

function main(L)

    # param_a = zeros(numParams)
    # param_b = similar(param_a)
    measure = Matrix{Float64}(undef, (length(param_rpe), 6))
    # measure = Matrix{Float64}(undef, (length(1:div(L, 2)), 10))
    # measureStd = Matrix{Float64}(undef, (length(1:div(L, 2)), 6))

    α = 1

    # ens_c = Dict("RPE" => 509.06, "PLE" => 56.10, "EXP" => 27.72) # r1, beta = 1
    # ens_c = Dict("RPE" => 123.03, "PLE" => 35.22, "EXP" => 24.31) # r2, beta = 1

    # e = "RPE"
    # for e in ["RPE","PLE","EXP"]
    #     for (i, l) in enumerate(1:div(L, 2))
    #         # defParamsPLE!(param_a, param_b, l, L-l)
    #         # print(param_a)
    #         # Y = getY(e, l, L-l)
    #         # print("τ = $(ens_c[e])\n")
    #         Y = getY(e, l, L-l)
    #         Y0 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    #         # Y0 = [1e-5, 5*1e-5, 1e-4, 5*1e-4, 1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1.0]
    #         # Y0 = 10 .^ range(log10(1e-5), stop=log10(1), length=10)
    #         # Y0 = [1e-3, 1e-2, 1e-1, 1, 10, 40.0]
    #         for (j, y) in enumerate(Y0)
    #             # print("y = $y \n")
    #             I = argmin(abs.(y .- Y))
    #             # print("$e $l => $I\n")
    #             if e == "RPE"
    #                 measure[i, j] = scalingLawRPE(param_rpe[I], l, L-l, α)
    #                 # measure[i, j] = participationEntropyRPE(param_rpe[I], l, L-l, α)
    #                 # measureStd[i, j] = scalingLawRPE(param_rpe[I], l, L-l, α)
    #             elseif e == "PLE"
    #                 measure[i, j] = scalingLawPLE(param_ple[I[1]], param_ple[I[2]], l, L-l, α)
    #                 # measure[i, j] = participationEntropyPLE(param_ple[I[1]], param_ple[I[2]], l, L-l, α)
    #                 # measureStd[i, j] = scalingLawPLE(param_ple[I[1]], param_ple[I[2]], l, L-l, α)
    #             else
    #                 measure[i, j] = scalingLawEXP(param_exp[I[1]], param_exp[I[2]], l, L-l, α)
    #                 # measure[i, j] = participationEntropyEXP(param_exp[I[1]], param_exp[I[2]], l, L-l, α)
    #                 # measureStd[i, j] = scalingLawEXP(param_exp[I[1]], param_exp[I[2]], l, L-l, α)
    #             end
    #         end
    #     end
    #     # writedlm("entropyScaling$(e)Y_R$α,L=$L,beta=$β.txt", measure)
    #     writedlm("participationEntropyScaling$(e)Y_S$α,L=$L,beta=$β.txt", measure)
    #     # writedlm("entropyScaling$(e)rescaledY_R1,L=$L,beta=$β.txt", measure)
    #     # writedlm("entropyStdScaling$(e)Y_R$α,L=$L,beta=$β.txt", measureStd)
    # end

    Y0 = [10, 100, 1e3, 1e4, 1e5, 1e6]

    Y = readdlm("comParamRPE,mu_X_la.txt")
    for (i, a) in enumerate(param_rpe)
        for (j, y) in enumerate(Y0)
            I = argmin(abs.(y .- Y[i, :]))
            measure[i, j] = scalingLawRPE(a, I, L-I, α)
        end
    end

    writedlm("entropyScaling$(e),mu,$α,L=$L,beta=$β.txt", measure)
end

# main(20)
# wishartMeasures(1, 1000, β)
# print(Threads.nthreads())
# print("\n")
# flush(stdout)

# l = 12
# for l in 10:10
# keepEigvals(expDecayCol, param_exp, param_exp, l, l, "exp")
# end
# l=11
# keepEigvals(powerLawCol, param_ple, param_ple, l, l, "ple")
# keepEigvals(rpeCol, param_rpe, l, l,"rpe")

# calcFlucs(param_rpe, l,"rpe")
# calcFlucs(param_ple, param_ple, l, "ple")
# calcFlucs(param_exp, param_exp, l, "exp")

# avgRhoMomentsWihtS1(expDecayCol, comParamExpCol, param_exp, param_exp, 2, l, l)
l = 10
checkApproxForR1(expDecayCol, param_exp, param_exp, l, l,"exp")
# app1 = checkApproxForR1(expDecayCol, param_exp, param_exp, l, l)
# app1 = checkApproxForR1(expDecayCol, param_exp, param_exp, l, l)
# writedlm("R1Approx,EXP_L=$l,two_param,beta=$β.txt", app1)
checkApproxForR1(powerLawCol, param_ple, param_ple, l, l, "ple")
# writedlm("R1Approx,PLE_L=$l,two_param,beta=$β.txt", app2)
checkApproxForR1(rpeCol, param_rpe, l, l,"rpe")
# writedlm("R1Approx,RPE_L=$l,two_param,beta=$β.txt", app3)

# l = 2
# for (i, j) in p
#     for b in 1:2
#         meas = vec(entDynamics(rpeCol, param_rpe, j, l, l, b))
#         writedlm("./data_final/Dynamics/RPE_L=$l,$i,beta=$b.txt", meas)
#     end
# end
# Y = vec(getY("EXP", 2, 1))

# using CairoMakie
# fig = Figure()
# ax = Axis(fig[1, 1])

# scatter!(ax, Y, meas_1, label="beta=1")
# scatter!(ax, Y, meas_2, label="beta=2")
# axislegend(position=:rc)

# print(maximum(meas_1), " ", maximum(meas_2))
