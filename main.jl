
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Ipopt
using Plots
using JuMP

VALE3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\VALE3.csv", DataFrame, select = [4])
AZUL4 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\AZUL4.csv", DataFrame, select = [4])
BBDC4 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\BBDC4.csv", DataFrame, select = [4])
ENEV3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\ENEV3.csv", DataFrame, select = [4])
ENGI11 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\ENGI11.csv", DataFrame, select = [4])
EQTL3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\EQTL3.csv", DataFrame, select = [4])
HYPE3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\HYPE3.csv", DataFrame, select = [4])
ITUB4 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\ITUB4.csv", DataFrame, select = [4])
JBSS3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\JBSS3.csv", DataFrame, select = [4])
LCAM3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\LCAM3.csv", DataFrame, select = [4])
LREN3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\LREN3.csv", DataFrame, select = [4])
MGLU3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\MGLU3.csv", DataFrame, select = [4])
MOVI3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\MOVI3.csv", DataFrame, select = [4])
NTCO3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\NTCO3.csv", DataFrame, select = [4])
PRIO3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\PRIO3.csv", DataFrame, select = [4])
RENT3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\RENT3.csv", DataFrame, select = [4])
TAEE11 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\TAEE11.csv", DataFrame, select = [4])
UNIP6 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\UNIP6.csv", DataFrame, select = [4])
VIVT3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\VIVT3.csv", DataFrame, select = [4])
VIIA3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\VIIA3.csv", DataFrame, select = [4])
PETR3 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\PETR3.csv", DataFrame, select = [4])
GOLL4 = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\GOLL4.csv", DataFrame, select = [4])
IBOV = CSV.read("C:\\Users\\User\\Desktop\\pythonProject1\\IBOV.csv", DataFrame, select = [7])
ACOES = ["VALE3", "AZUL4", "BBDC4", "ENEV3", "ENGI11", "EQTL3", "HYPE3", "ITUB4", "JBSS3", "LCAM3", "LREN3", "MGLU3", "MOVI3", "NTCO3", "PRIO3", "RENT3", "TAEE11", "UNIP6", "VIVT3", "VIIA3", "PETR3", "GOLL4"]
function conv(df)
    Aux = zeros(length(df[:,1]))
    for i in 1:length(df[:,1])
        Aux[i] = parse(Float64, replace(df[i,1], "," => "."))
    end
    return Aux
end
function conv2(df)
    Aux = zeros(length(df[:,1]))
    for i in 1:length(df[:,1])

        Aux[i] = parse(Float64, replace(chop(df[i,1], head = 0, tail = 1), "," => "."))
        Aux[i] = Aux[i]/100
    end
    return Aux  
end
#matriz com a variação de cada ativo
D = zeros(length(VALE3[:,1]), length(ACOES))
D[:,1] = conv(VALE3)
D[:,2] = conv(AZUL4)
D[:,3] = conv(BBDC4)
D[:,4] = conv(ENEV3)
D[:,5] = conv(ENGI11)
D[:,6] = conv(EQTL3)
D[:,7] = conv(HYPE3)
D[:,8] = conv(ITUB4)
D[:,9] = conv(JBSS3)
D[:,10] = conv(LCAM3)
D[:,11] = conv(LREN3)
D[:,12] = conv(MGLU3)
D[:,13] = conv(MOVI3)
D[:,14] = conv(NTCO3)
D[:,15] = conv(PRIO3)
D[:,16] = conv(RENT3)
D[:,17] = conv(TAEE11)
D[:,18] = conv(UNIP6)
D[:,19] = conv(VIVT3)
D[:,20] = conv(VIIA3)
D[:,21] = conv(PETR3)
D[:,22] = conv(GOLL4)

#variação do mercado
Dm = zeros(length(IBOV[:,1]))
Dm[:,1] = conv2(IBOV)


#matriz com os retornos médios dos ativos, sendo, a primeira => GOLL4, segunda linha => PETR3, terceira linha => OIBR3, quarta linha => VALE3
R = mean(D, dims = 1)[:]
Rf = 10/(100*12) #peguei valor de um título préfixado com 10% a.a
#variação média do mercado
Rm = float(mean(Dm, dims = 1))
#matriz de Covariância entre os ativos:
Q = cov(D)
"""Podemos colocar corrected = false para utilizar todos os n períodos, mas na prática geralmente usamos n-1 
períodos. (vai de pessoa para pessoa). Note também que, na matriz de covariâncias, a diagonal é a variação do
ativo. Portanto, tirando a raíz da diagonal obtemos o risco de cada ativo. 
"""

#Matriz com as ações, o retorno médio e a variância das ações
Matriz_resumo = [ACOES R sqrt.(diag(Q))]
print(Matriz_resumo)

#Modelo de Markowitz sem venda a descoberto e sem ativo livre de risco
function MarkowitzSVSA(Q, λ = 0.0)
    m = length(R)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:m] >= 0)
    @objective(model, Min, dot(x, Q*x))
    @constraint(model, sum(x) == 1)
    optimize!(model)
    X = value.(x)
    return X
end
#função que retorna a carteira
function carteira(x, Q)
    return sqrt(x'*Q*x), dot(R, x)
end
#aplicando

X1 = MarkowitzSVSA(Q)
σc1, R1 = carteira(X1, Q)
print("O risco e o retorno da carteira um são: σc1 = $σc1, R1 = $R1 ")


## Agora vamos fazer os plots!!
plot(D, leg = false)
retorno_tempo_1 = X1'*D'
plot(retorno_tempo_1', leg = false)
print(retorno_tempo_1)
#ganho cumulativo das ações
ganho_cumulativo = cumprod(1 .+ D', dims =1)
ganho_cumulativo = [1.0; ganho_cumulativo]
plot!(ganho_cumulativo, leg = false)
#ganho cumulativo da carteira (dar uma olhada pq ta meio estranho)
ganho_cumulativo_carteira = cumprod(1 .+ retorno_tempo_1', dims =1)
ganho_cumulativo_carteira = [1.0; ganho_cumulativo_carteira]
plot!(ganho_cumulativo, c =:green, leg = false)

"""

Aqui terminamos a primeira carteira, sem venda a descoberto e sem at.liv. de risco e começamos o modelo 2
com venda a descoberto e sem ativo livre de risco

"""
function MarkowitzCVSA(Q, λ = 0.0)
    m = length(R)
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    model = Model(optimizer)
    @variable(model, x[1:m])
    @objective(model, Min, dot(x, Q*x))
    @constraint(model, sum(x) == 1)
    optimize!(model)
    X2 = value.(x)
    return X2
end
#aplicando
X2 = MarkowitzCVSA(Q)
σc2, R2 = carteira(X2, Q)
print("O Retorno e Risco da carteira X2 são: σc2 = $σc2, R2 = $R2\n")
print("A proporção X2 para cada ativo é:", X2)
retorno_tempo_2= X2'*D'
print(retorno_tempo_2)

ganho_cumulativo_carteira_2 = cumprod(retorno_tempo_2', dims = 1 )
ganho_cumulativo_carteira_2 = [ganho_cumulativo_carteira_2]
plot(ganho_cumulativo_carteira_2)
"""
Aqui nós terminamos o modelo dois, com venda a descoberto e sem ativo livre de risco e começamos o modelo 3
sem venda a descoberto mas com ativo livre de risco

"""


function MarkowitzSVCA(Q, λ= 0.0)
    m = length(R)
    Rf = 10/(100*12)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:m] >= 0)
    @expression(model, expr1, R' * x)
    @expression(model, expr2, x'*Q*x)
    @NLobjective(model, Max, ((expr1 - Rf)/sqrt(expr2)))
    @constraint(model, sum(x) == 1)
    optimize!(model)
    return value.(x)
end
#aplicando
X3 = MarkowitzSVCA(Q)

σc3, Rc3= carteira(X3, Q)
print("O Retorno e o risco da carteira X3 são: σc3 = $σc3, Rc3 = $Rc3." )
print("A proporção X3 para cada ativo é:", X3)
retorno_tempo_3= X3'*D'
print(retorno_tempo_3)

ganho_cumulativo_carteira_3 = cumprod(retorno_tempo_3', dims = 1 )
ganho_cumulativo_carteira_3 = [ganho_cumulativo_carteira_3]
plot(ganho_cumulativo_carteira_3)


"""
AQUI TERMINA O MODELO DE MARCOS UITZ COM ATIVO LIVRE DE RISCO MAS SEM VENDA A DESCOBERTO E COMEÇA O MODELO
COM ATIVO LIVRE DE RISCO E COM VENDA A DESCOBERTO
"""
#Resolução por sistema linear. Temos que Qz = R - Rf
Rf = mean(Ativo_livre)
RF = ones(length(ACOES))
X4 = zeros(length(ACOES))
RF = Rf*RF
z = Q\(R -RF)
for i in 1:length(ACOES)
    X4[i] = z[i]/sum(z)
end

σc4, Rc4= carteira(X4, Q)
print("O Retorno e o risco da carteira X4 são: σc4 = $σc4, Rc4 = $Rc4.")
print("A proporção X4 para cada ativo é:", X4)
retorno_tempo_4 = X4'*D'
print(retorno_tempo_4)

ganho_cumulativo_carteira_4 = cumprod(retorno_tempo_4', dims = 1 )
ganho_cumulativo_carteira_4 = [ganho_cumulativo_carteira_4]
plot(ganho_cumulativo_carteira_4)


"""

MODELO ÍNDICE ÚNICO

"""

#Definindo R médio das ações no tempo t (pode ser útil para analisar)

Rit = zeros(length(D[:,1]), length(ACOES))
local_aux = zeros(length(D[:,1]))

for i in range(1,length(D[1,:]))
    
    for t in range(1,length(D[:,1]))
        aux = zeros(t)
        
        for j in range(1,t)
            local_aux[j] = D[j,i]
        end

        
        Rit[t,i] = mean(local_aux)
    end
end

#cálculo de alpha e beta
σim = zeros(length(ACOES)) #covariância da ação i com o mercado
βi = zeros(length(ACOES)) #beta da ação i
αi = zeros(length(ACOES)) #alpha da ação i
σm = var(Dm) #variância do mercado analisado 
Rm = Rm[1] 
for i in 1:length(ACOES)
    σim[i] = cov(Dm, D[:,i])
    βi[i] = σim[i]/σm
    αi[i] = R[i] - βi[i]*Rm
end

#Variância dos erros
σe = zeros(length(ACOES))
for i in 1:length(ACOES)
    S = 0.0
    for t in 1: length(D[:,1])
        S += (D[t,i] - (αi[i] +βi[i]*Dm[t]))^2
    end
    σe[i] = S/length(D[:,1])
end
σe
#covariância dos ativos

σij = zeros(length(ACOES),length(ACOES))
for i in 1:length(ACOES)
    for j in 1:length(ACOES)
        if i == j
            σij[i,j] = βi[i]*βi[j]*σm + σe[i]
        else
            σij[i,j] = βi[i]*βi[j]*σm
        end  
    end
end

#Definindo o X da carteira de índice único sem venda a descoberto
Xiu1 = MarkowitzSVSA(σij)


#alpha e beta da carteira

βc1 = 0
αc1 = 0

for i in 1:length(Xiu1)
    αc1 = αc1 + Xiu1[i]*αi[i]
    βc1 = βc1 + βi[i]*Xiu1[i]
end
#Retorno pela fórmula
Rciu1 = αc1 +βc1*Rm
print("O Retorno e o Beta da carteira Xiu1 são: βc1 = $βc1, Riu1 = $Rciu1 .")
print("A proporção Xiu1 para cada ativo é:", Xiu1)
retorno_tempo_iu1 = Xiu1'*D'
print(retorno_tempo_iu1)

ganho_cumulativo_carteira_iu1 = cumprod(retorno_tempo_iu1', dims = 1 )
ganho_cumulativo_carteira_iu1 = [1.0; ganho_cumulativo_carteira_iu1]
plot(ganho_cumulativo_carteira_iu1)

 #carteira de índice único com venda a descoberto
Xiu2 = MarkowitzCVSA(σij)
sqrt(Xiu2' * σij * Xiu2)
#alpha e beta da carteira
βc2 = 0
αc2 = 0
for i in 1:length(Xiu2)
    αc2 += Xiu2[i]*αi[i]
    βc2 += βi[i]*Xiu2[i]
end
#Retorno pela fórmula
Rciu2 = αc2 + βc2*Rm
print("O Retorno e o Beta da carteira Xiu2 são: βc2 = $βc2, Riu2 = $Rciu2.")
print("A proporção Xiu1 para cada ativo é:", Xiu2)

#Retorno por tempo de cada ação
retorno_tempo_iu2 = Xiu2'*D'
print(retorno_tempo_iu2)

#Retorno comulativo da carteira
ganho_cumulativo_carteira_iu2 = cumprod(retorno_tempo_iu2', dims = 1 )
ganho_cumulativo_carteira_iu2 = [1.0; ganho_cumulativo_carteira_iu2]
plot(ganho_cumulativo_carteira_iu2)









#Variância individual de cada ação.
σ = zeros(length(ACOES))
auxiliar = zeros(length(ACOES))
for i in 1:length(R)


    for j in 1:length(D[:,1])
        auxiliar[i] += (D[j,i] - R[i])^2

    end
    σ[i] = auxiliar[i]/length(D[:,1])
    
end

#Pode ser calculada também da seguinte maneira (bem mais compacta)
Var = diag(Q)
Desvio_padrão = sqrt.(Var)