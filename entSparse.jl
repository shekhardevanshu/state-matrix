using QuantumInformation
using LinearAlgebra
using Statistics
using Distributions
using FLoops
using DelimitedFiles
using SparseArrays
BLAS.set_num_threads(1)

function hamming_distance(num1::Int, num2::Int)
    differing_bits = num1 ⊻ num2
    count = 0
    while differing_bits != 0
        differing_bits &= differing_bits - 1
        count += 1
    end
    return count
end

function hdMat(N1, N2)
    A = zeros(N1, N2)
    for i in 1:N2
        for j in 1:N1
            A[j, i] = hamming_distance(i-1, j-1)
        end
    end
    return A
end

function sparseEns(N1::Integer, N2::Integer, nn, params::Dict)
    A = zeros(ComplexF64, N1, N2)
    # nn = 1
    w, ws = params["w"], params["ws"]
    A[:,1] = rand(Normal(), N1) + im*rand(Normal(), N1)
    # H = readdlm("hamming_distances.txt")
    # dist2 = Normal(0., params["w"])
    for i=2:N2 # column index
        # if i == 1
        # dist1 = Normal(0., w)
        # dist2 = Normal(0., ws)
        # else
        # dist1 = Normal(0., sqrt(exp(-((i-1)/(w))^2)))
        # dist2 = Normal(0., sqrt(exp(-((i-1)/(ws))^2)))
            # dist1 = Normal(0., (t*w)/(i-1))
            # dist2 = Normal(0., (t*ws)/(i-1))
        # end

        for j=1:N1 # row index
            dist1 = Normal(0., sqrt(exp(-((i-1)*j/(w)^2))))
            dist2 = Normal(0., sqrt(exp(-((i-1)*j/(ws)^2))))
            z = hamming_distance(i-1, j-1)
            # z = H[j, i]
            if z == 0
                A[j, i] = rand(dist1) + im*rand(dist1)
            elseif z <= nn
                A[j, i] = rand(dist2) + im*rand(dist2)
            end
        end
    end
    
    return A
end

function sparseY(N1::Integer, N2::Integer, nn, params::Dict)
    
    # w, ws, t = params["w"], params["ws"], params["t"]
    w, ws = params["w"], params["ws"]
    gamma = 0.25
    # if w != 0 || ws != 0
    #     gamma = 0.4 / max(w^2, ws^2) # for y >= 0
    # end
    M = N1*N2
    # H = readdlm("hamming_distances.txt")
    # h1 = w^2
    # h2 = ws^2
    
    #initial condition
    # y = N1*log(abs(1 - 2*gamma))
    y=0

    for i=2:N2

        # h1 = exp(-((i-1)/(w))^2)
        # h2 = exp(-((i-1)/(ws))^2)
        # @show y
        for j=1:N1
            h1 = exp(-((i-1)*j/(w)^2))
            h2 = exp(-((i-1)*j/(ws)^2))
            z = hamming_distance(i-1, j-1)
            # z = H[j, i]
            if z == 0
                y += log(abs(1 - 2*gamma*h1))
                # M += 1
            elseif z <= nn
                y += log(abs(1 - 2*gamma*h2))
                # M += 1
            end
        end

    end

    y *= (-1/(2*M*gamma))
end


function main(args)
    L1 = parse(Int, args[1])
    # L2 = parse(Int, args[2])
    # L2 = 20-L1
    L2 = L1
    # w = parse(Float32, args[2])
    # ws = parse(Float32, args[3])
    nn = parse(Int, args[2])
    samples = 1000

    N1 = 2^L1; N2 = 2^L2
    # t_final = 4*1200
    # time = LinRange(0, t_final ^(1/3), 200).^3
    w_arr = LinRange(0, 200 ^(1/3), 2) .^3
    ws_arr = LinRange(0, 2500 ^(1/3), 3) .^3
    # writedlm("hamming_distances.txt", hdMat(N1, N2))
    # Entsp = zeros(length(time))
    Entsp = []
    # Y = similar(time)
    Y = []
    # for i in range(0., step=0.2, stop=1.0)
    i = 1
    for ws in ws_arr
        for w in w_arr

            # p = Dict("w"=>w, "ws"=>ws,"t"=>t)
            p = Dict("w"=>w, "ws"=>ws)
            ent = zeros(samples)
            @floop for i=1:samples
                A = sparseEns(N1, N2, nn, p)
                rho = A*A'
                rho ./= tr(rho)
                ent[i] = vonneumann_entropy(rho)/log(2)
                # ent[i] = -log2(tr(rho^2))
            end
            # Entsp[i] = mean(ent)
            push!(Entsp, mean(ent))
            print(mean(ent))
            # Y[i] = sparseY(N1, N2, nn, p)
            push!(Y, sparseY(N1, N2, nn, p))
            # @show sparseY(N1, N2, nn, p)
            print("i = $i \n")
            flush(stdout)
            i += 1
        end
    end
    # data = readdlm("./sparseEnsemble/avg_vn_sparse,l1=$L1,l2=$L2,w=$w,ws=$ws,nn=$nn.txt")
    # data = readdlm("./sparseEnsemble/avg_vn_sparse,l1=$L1,l2=$L2,nn=$nn,beta=2.txt")
    # Entsp = data[:, 1]

    # writedlm("./sparseEnsemble/avg_vn_sparse,l1=$L1,l2=$L2,w=$w,ws=$ws,nn=$nn,beta=2.txt", [Entsp (Y .- Y[1])])
    writedlm("./sparseEnsemble/avg_vn_sparse,l1=$L1,l2=$L2,nn=$nn,beta=2.txt", [Entsp (Y .- Y[1])])
    # writedlm("./sparseEnsemble/avg_r2_sparse,l1=$L1,l2=$L2,w=$w,ws=$ws,nn=$nn,beta=2.txt", [Entsp (Y .- Y[1])])
    # writedlm("y_AE2D,w=$w,ws=$ws.txt", Y)
end

main(ARGS)
