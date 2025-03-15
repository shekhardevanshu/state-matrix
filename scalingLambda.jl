using DelimitedFiles
using LinearAlgebra
using FLoops
using Statistics, Distributions
# using QuantumInformation
BLAS.set_num_threads(1)

function expDecayCol(L1::Integer, L2::Integer, a, β::Integer)
    d1, d2 = 2^L1, 2^L2
    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(exp(-((j-1)*i/a))))
                A[i, j] = rand(dist)
            end 
        end

    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(exp(-((j-1)*i/a))))
                A[i, j] = rand(dist) + im*rand(dist)
            end 
        end
    end
    return A
end

function powerLawCol(L1::Integer, L2::Integer, a::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2

    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(1/(1 + ((j-1)*i/a))))
                A[i, j] = rand(dist)
            end 
        end

    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        for j=2:d2
            for i=1:d1
                dist = Normal(0., sqrt(1/(1 + ((j-1)*i/a))))
                A[i, j] = rand(dist) + im*rand(dist)
            end 
        end
    end

    return A
end

function rpeCol(L1::Integer, L2::Integer, a::Real, β::Integer)
    d1, d2 = 2^L1, 2^L2

    if β == 1
        A = Matrix{Float64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(1 + a)))
        A[:, 2:d2] = rand(dist, (d1, d2-1))
    elseif β == 2
        A = Matrix{ComplexF64}(undef, (d1, d2))
        A[:,1] = rand(Normal(), d1) + im*rand(Normal(), d1)
        dist = Normal(0., sqrt(1/(1 + a)))
        A[:, 2:d2] = rand(dist, (d1, d2-1)) + im*rand(dist, (d1, d2-1))
    end

    return A
end

# function hamming_distance(num1::Int, num2::Int)
#     differing_bits = num1 ⊻ num2
#     count = 0
#     while differing_bits != 0
#         differing_bits &= differing_bits - 1
#         count += 1
#     end
#     return count
# end

# function sparseEns(N1::Integer, N2::Integer, nn, w, ws)
#     A = zeros(ComplexF64, N1, N2)
#     # nn = 1
#     A[:,1] = rand(Normal(), N1) + im*rand(Normal(), N1)
#     # H = readdlm("hamming_distances.txt")
#     # dist2 = Normal(0., params["w"])
#     for i=2:N2 # column index
#         # if i == 1
#         # dist1 = Normal(0., w)
#         # dist2 = Normal(0., ws)
#         # else
#         # dist1 = Normal(0., sqrt(exp(-((i-1)/(w))^2)))
#         # dist2 = Normal(0., sqrt(exp(-((i-1)/(ws))^2)))
#             # dist1 = Normal(0., (t*w)/(i-1))
#             # dist2 = Normal(0., (t*ws)/(i-1))
#         # end

#         for j=1:N1 # row index
#             dist1 = Normal(0., sqrt(exp(-((i-1)*j/(w)^2))))
#             dist2 = Normal(0., sqrt(exp(-((i-1)*j/(ws)^2))))
#             z = hamming_distance(i-1, j-1)
#             # z = H[j, i]
#             if z == 0
#                 A[j, i] = rand(dist1) + im*rand(dist1)
#             elseif z <= nn
#                 A[j, i] = rand(dist2) + im*rand(dist2)
#             end
#         end
#     end
    
#     return A
# end

# function entropyWtrace(ensemble::Function, mu::Float64, a::Real, L1::Integer, L2::Integer, samples::Integer, ens)
#     s1 = LinRange(0.1, 1, 50)
#     measure = zeros(length(s1))
#     ent = zeros(samples)

#     for (i, t) in enumerate(s1)
#         @floop for k in 1:samples
#             C = ensemble(L1, L2, mu, 2)
#             rho = C*C'
#             rho = t * rho / tr(rho)
#             eVals = eigvals(rho)
#             if a == 1
#                 ent[k] = -sum([e*log2(e) for e in eVals])
#             elseif a == 0
#                 ent[k] = -sum([log2(e) for e in eVals])
#             elseif a == 2
#                 ent[k] = -log2(sum([e^2 for e in eVals]))
#             end
#         end
#         measure[i] = mean(ent)
#     end

#     writedlm("entropy,n=$a,vsS_1,$ens.txt", measure)
# end

function EigvalsCrit(ensemble::Function, mu, c, L1::Integer, L2::Integer, samples::Integer, ens)

    v = zeros(samples, 2^L1)
    C = zeros(ComplexF64, 2^L1, 2^L2)
    rho = similar(C)

    for (i, a) in enumerate(mu)
        try
            V = readdlm("./evalsCrit/evals_crit_2,c=$c,alpha=$(round(a, digits=3)),l=$L1,$ens.txt")
        catch
            for k in 1:samples
                C .= ensemble(L1, L2, c*2^(L1*a), 2)
                mul!(rho, C, C')
                rho ./= tr(rho)
                v[k, :] = abs.(eigvals(rho))

                # if k%10 == 0
                #     print("iteration $k \n")
                #     flush(stdout)
                # end

            end
            writedlm("./evalsCrit/evals_crit_2,c=$c,alpha=$(round(a, digits=3)),l=$L1,$ens.txt", v)
            print("completed i = $i \n")
            flush(stdout)
        end
    end

end

# function R0crit(ensemble::Function, mu, c, L1::Integer, L2::Integer, samples::Integer, ens)
#     measure = zeros(length(mu))
#     ent = zeros(samples)

#     for (i, a) in enumerate(mu)
#         @floop for k in 1:samples
#             C = ensemble(L1, L2, c*2^(L1*a), 2)
#             rho = C*C'
#             rho = rho / tr(rho)
#             eVals = eigvals(rho)
#             ent[k] = -sum([log2(e) for e in eVals])
#         end
#         measure[i] = mean(ent)
#         print("completed i = $i \n")
#         flush(stdout)
#     end

#     writedlm("r0critical,c=$c,l=$L1,$ens.txt", [mu measure])
# end

# function R1crit(ensemble::Function, mu, c, L1::Integer, L2::Integer, samples::Integer, ens)
#     measure = zeros(length(mu))
#     ent = zeros(samples)
#     C = zeros(ComplexF64, (2^L1, 2^L2))
#     rho = similar(C)
#     eVals = zeros(2^L1)

#     for (i, a) in enumerate(mu)
#         for k in 1:samples
#             C .= ensemble(L1, L2, c*2^(L1*a), 2)
#             mul!(rho, C, C')
#             rho ./= tr(rho)
#             eVals .= abs.(eigvals(rho))
#             ent[k] = -sum([e*log2(e) for e in eVals])
#             print("completed iteration $k \n")
#             flush(stdout)
#         end
#         measure[i] = mean(ent)
#         print("---------------------")
#         print("completed i = $i \n")
#         flush(stdout)
#     end

#     writedlm("r1critical,c=$c,l=$L1,$ens.txt", [mu measure])
# end

function Rt(V)
    t = 0

    for e1 in V
        for e2 in V
            if e1 != e2
                t += e1*log2(e1) / (e1 - e2)
            end
        end
    end

    t
end

function Rncrit(alpha, c, L1::Integer, ens)

    r1 = zeros(length(alpha))
    r0 = similar(r1)
    # rt = similar(r1)
    
    @floop for (i, a) in enumerate(round.(alpha, digits=3))
        eVals = readdlm("./evalsCrit/evals_crit,c=$c,alpha=$a,l=$L1,$ens.txt")
        # eVals = readdlm("./evalsCrit/evals_crit_2,c=$c,alpha=$a,l=$L1,$ens.txt")
        @show samples = size(eVals, 1)
        # r1[i] = mean([-sum([e*log2(e) for e in eVals[k, :]]) for k in 1:samples])
        r0[i] = mean([-sum([log2(e) for e in eVals[k, :]]) for k in 1:samples])
        # rt[i] = mean([Rt(eVals[k, :]) for k in 1:samples])
        
        print("completed i = $i \n")
        flush(stdout)
    end

    # writedlm("r1critical_2,c=$c,l=$L1,$ens.txt", [alpha r1])
    writedlm("r0critical,c=$c,l=$L1,$ens.txt", [alpha r0])
    # writedlm("rtcritical,c=$c,l=$L1,$ens.txt", [alpha rt])

end


function main()
    vn = zeros(10)
    ens = "EE"
    # Y = ["10", "100", "1000", "10000", "100000", "1e6"]
    Y = ["0.001", "0.01", "0.1", "1", "10"]
    # Y = ["10"]
    # nns = [3, 4, 4, 6, 6, 12]
    # w_arr = [10, 15, 100, 100, 300, 500]
    # Y = ["10"]
    for (i, y) in enumerate(Y)
        # a_ee = readdlm("alpha_EE,Y=$y.txt")
        # a_ee = readdlm("alpha_PE,D1,Y=$y.txt")
        a_ee = readdlm("alpha_$ens,D1,Y=$y.txt")
        sampleSize = 1000
        for l in 1:10
            a = 2^((2-a_ee[l])*l) * 0.1
            # a = 0.1 * 2^(a_ee[l]*l) # BE
            ent = zeros(sampleSize)
            @floop for s in 1:sampleSize
                # A = powerLawCol(l, 20-l, a, 2)
                A = expDecayCol(l, 20-l, a, 2)
                # A = rpeCol(l, 20-l, a, 2)
                # A = sparseEns(2^l, 2^(20-l), 6, 500, a_ee[l])
                rho = A*A'
                rho ./= tr(rho)
                # ent[s] = vonneumann_entropy(rho)/log(2)
                ent[s] = -sum([e*log2(e) for e in abs.(eigvals(rho))])
                # ent[s] = renyi_entropy(rho, 2)/log(2)
            end
            @show vn[l] = mean(ent)
        end

        # writedlm("r2_avg,EE,Y=$y.txt", vn)
        writedlm("vn_avg,$ens,Yent=$y.txt", vn)
    end
    # l=10
    # y = 1000
    # alpha = readdlm("alpha_BE,Y=$y.txt")
    # a = 2^((alpha[l])*l) * 0.1
    # for n in 0:2
    #     entropyWtrace(rpeCol, a, n, l, l, 1000, "BE")
    # end

    # alpha = readdlm("alpha_PE,Y=$y.txt")
    # a = 2^((2-alpha[l])*l) * 0.1
    # for n in 0:2
    #     entropyWtrace(powerLawCol, a, n, l, l, 1000, "PE")
    # end

    # alpha = readdlm("alpha_EE,Y=$y.txt")
    # a = 2^((2-alpha[l])*l) * 0.1
    # for n in 0:2
    #     entropyWtrace(expDecayCol, a, n, l, l, 1000, "EE")
    # end

    # samples = [1000, 500, 100, 20, 10]
    # samples = [1000, 500, 50, 10]
    # samples = [1, 1, 1, 1]
    
    # ================= BE

    # alpha = LinRange(1.1,2,15)
    # c=0.1
    # ens = "BE"

    # for (l, s) in zip(10:13, samples)
    #     EigvalsCrit(rpeCol, alpha, c, l, l, s, ens)
    #     print("===================\n")
    #     print("completed L = $l \n")
    #     flush(stdout)
    # end

    # for l in 10:13
    #     Rncrit(alpha, c, l, ens)
    # end

    # ================= PE

    # alpha = LinRange(0.,0.9,15)
    # c=1
    # ens = "PE"

    # for (l, s) in zip(10:13, samples)
    #     EigvalsCrit(powerLawCol, alpha, c, l, l, s, ens)
    #     Rncrit(alpha, c/log(l), l, ens)
    #     print("===================\n")
    #     print("completed L = $l \n")
    #     flush(stdout)
    # end

    # for l in 10:13
    #     Rncrit(alpha, c, l, ens)
    # end

    
    # ================= EE
    # alpha = LinRange(0.,0.9,15)
    # c = 7
    # ens = "EE"

    # for (l, s) in zip(10:13, samples)
    #     EigvalsCrit(expDecayCol, alpha, c, l, l, s, ens)
    #     print("===================\n")
    #     print("completed L = $l \n")
    #     flush(stdout)
    # end

    # for l in 10:13
    #     Rncrit(alpha, c, l, ens)
    # end
    
end

main()
