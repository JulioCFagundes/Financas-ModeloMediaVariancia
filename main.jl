
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
#matriz com a varia????o de cada ativo
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

#varia????o do mercado
Dm = zeros(length(IBOV[:,1]))
Dm[:,1] = conv2(IBOV)


#matriz com os retornos m??dios dos ativos, sendo, a primeira => GOLL4, segunda linha => PETR3, terceira linha => OIBR3, quarta linha => VALE3
R = mean(D, dims = 1)[:]
Rf = 10/(100*12) #peguei valor de um t??tulo pr??fixado com 10% a.a
#varia????o m??dia do mercado
Rm = float(mean(Dm, dims = 1))
#matriz de Covari??ncia entre os ativos:
Q = cov(D)
"""Podemos colocar corrected = false para utilizar todos os n per??odos, mas na pr??tica geralmente usamos n-1 
per??odos. (vai de pessoa para pessoa). Note tamb??m que, na matriz de covari??ncias, a diagonal ?? a varia????o do
ativo. Portanto, tirando a ra??z da diagonal obtemos o risco de cada ativo. 
"""

#Matriz com as a????es, o retorno m??dio e a vari??ncia das a????es
Matriz_resumo = [ACOES R sqrt.(diag(Q))]
print(Matriz_resumo)

#Modelo de Markowitz sem venda a descoberto e sem ativo livre de risco
function MarkowitzSVSA(Q, ?? = 0.0)
    m = length(R)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:m] >= 0)
    @objective(model, Min, dot(x, Q*x))
    @constraint(model, sum(x) == 1)
    optimize!(model)
    X = value.(x)
    return X
end
#fun????o que retorna a carteira
function carteira(x, Q)
    return sqrt(x'*Q*x), dot(R, x)
end
#aplicando

X1 = MarkowitzSVSA(Q)
??c1, R1 = carteira(X1, Q)
print("O risco e o retorno da carteira um s??o: ??c1 = $??c1, R1 = $R1 ")


## Agora vamos fazer os plots!!
plot(D, leg = false)
retorno_tempo_1 = X1'*D'
plot(retorno_tempo_1', leg = false)
print(retorno_tempo_1)
#ganho cumulativo das a????es
ganho_cumulativo = cumprod(1 .+ D', dims =1)
ganho_cumulativo = [1.0; ganho_cumulativo]
plot!(ganho_cumulativo, leg = false)
#ganho cumulativo da carteira (dar uma olhada pq ta meio estranho)
ganho_cumulativo_carteira = cumprod(1 .+ retorno_tempo_1', dims =1)
ganho_cumulativo_carteira = [1.0; ganho_cumulativo_carteira]
plot!(ganho_cumulativo, c =:green, leg = false)

"""

Aqui terminamos a primeira carteira, sem venda a descoberto e sem at.liv. de risco e come??amos o modelo 2
com venda a descoberto e sem ativo livre de risco

"""
function MarkowitzCVSA(Q, ?? = 0.0)
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
??c2, R2 = carteira(X2, Q)
print("O Retorno e Risco da carteira X2 s??o: ??c2 = $??c2, R2 = $R2\n")
print("A propor????o X2 para cada ativo ??:", X2)
retorno_tempo_2= X2'*D'
print(retorno_tempo_2)

ganho_cumulativo_carteira_2 = cumprod(retorno_tempo_2', dims = 1 )
ganho_cumulativo_carteira_2 = [ganho_cumulativo_carteira_2]
plot(ganho_cumulativo_carteira_2)
"""
Aqui n??s terminamos o modelo dois, com venda a descoberto e sem ativo livre de risco e come??amos o modelo 3
sem venda a descoberto mas com ativo livre de risco

"""


function MarkowitzSVCA(Q, ??= 0.0)
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

??c3, Rc3= carteira(X3, Q)
print("O Retorno e o risco da carteira X3 s??o: ??c3 = $??c3, Rc3 = $Rc3." )
print("A propor????o X3 para cada ativo ??:", X3)
retorno_tempo_3= X3'*D'
print(retorno_tempo_3)

ganho_cumulativo_carteira_3 = cumprod(retorno_tempo_3', dims = 1 )
ganho_cumulativo_carteira_3 = [ganho_cumulativo_carteira_3]
plot(ganho_cumulativo_carteira_3)


"""
AQUI TERMINA O MODELO DE MARCOS UITZ COM ATIVO LIVRE DE RISCO MAS SEM VENDA A DESCOBERTO E COME??A O MODELO
COM ATIVO LIVRE DE RISCO E COM VENDA A DESCOBERTO
"""
#Resolu????o por sistema linear. Temos que Qz = R - Rf
Rf = mean(Ativo_livre)
RF = ones(length(ACOES))
X4 = zeros(length(ACOES))
RF = Rf*RF
z = Q\(R -RF)
for i in 1:length(ACOES)
    X4[i] = z[i]/sum(z)
end

??c4, Rc4= carteira(X4, Q)
print("O Retorno e o risco da carteira X4 s??o: ??c4 = $??c4, Rc4 = $Rc4.")
print("A propor????o X4 para cada ativo ??:", X4)
retorno_tempo_4 = X4'*D'
print(retorno_tempo_4)

ganho_cumulativo_carteira_4 = cumprod(retorno_tempo_4', dims = 1 )
ganho_cumulativo_carteira_4 = [ganho_cumulativo_carteira_4]
plot(ganho_cumulativo_carteira_4)


"""

MODELO ??NDICE ??NICO

"""

#Definindo R m??dio das a????es no tempo t (pode ser ??til para analisar)

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

#c??lculo de alpha e beta
??im = zeros(length(ACOES)) #covari??ncia da a????o i com o mercado
??i = zeros(length(ACOES)) #beta da a????o i
??i = zeros(length(ACOES)) #alpha da a????o i
??m = var(Dm) #vari??ncia do mercado analisado 
Rm = Rm[1] 
for i in 1:length(ACOES)
    ??im[i] = cov(Dm, D[:,i])
    ??i[i] = ??im[i]/??m
    ??i[i] = R[i] - ??i[i]*Rm
end

#Vari??ncia dos erros
??e = zeros(length(ACOES))
for i in 1:length(ACOES)
    S = 0.0
    for t in 1: length(D[:,1])
        S += (D[t,i] - (??i[i] +??i[i]*Dm[t]))^2
    end
    ??e[i] = S/length(D[:,1])
end
??e
#covari??ncia dos ativos

??ij = zeros(length(ACOES),length(ACOES))
for i in 1:length(ACOES)
    for j in 1:length(ACOES)
        if i == j
            ??ij[i,j] = ??i[i]*??i[j]*??m + ??e[i]
        else
            ??ij[i,j] = ??i[i]*??i[j]*??m
        end  
    end
end

#Definindo o X da carteira de ??ndice ??nico sem venda a descoberto
Xiu1 = MarkowitzSVSA(??ij)


#alpha e beta da carteira

??c1 = 0
??c1 = 0

for i in 1:length(Xiu1)
    ??c1 = ??c1 + Xiu1[i]*??i[i]
    ??c1 = ??c1 + ??i[i]*Xiu1[i]
end
#Retorno pela f??rmula
Rciu1 = ??c1 +??c1*Rm
print("O Retorno e o Beta da carteira Xiu1 s??o: ??c1 = $??c1, Riu1 = $Rciu1 .")
print("A propor????o Xiu1 para cada ativo ??:", Xiu1)
retorno_tempo_iu1 = Xiu1'*D'
print(retorno_tempo_iu1)

ganho_cumulativo_carteira_iu1 = cumprod(retorno_tempo_iu1', dims = 1 )
ganho_cumulativo_carteira_iu1 = [1.0; ganho_cumulativo_carteira_iu1]
plot(ganho_cumulativo_carteira_iu1)

 #carteira de ??ndice ??nico com venda a descoberto
Xiu2 = MarkowitzCVSA(??ij)
sqrt(Xiu2' * ??ij * Xiu2)
#alpha e beta da carteira
??c2 = 0
??c2 = 0
for i in 1:length(Xiu2)
    ??c2 += Xiu2[i]*??i[i]
    ??c2 += ??i[i]*Xiu2[i]
end
#Retorno pela f??rmula
Rciu2 = ??c2 + ??c2*Rm
print("O Retorno e o Beta da carteira Xiu2 s??o: ??c2 = $??c2, Riu2 = $Rciu2.")
print("A propor????o Xiu1 para cada ativo ??:", Xiu2)

#Retorno por tempo de cada a????o
retorno_tempo_iu2 = Xiu2'*D'
print(retorno_tempo_iu2)

#Retorno comulativo da carteira
ganho_cumulativo_carteira_iu2 = cumprod(retorno_tempo_iu2', dims = 1 )
ganho_cumulativo_carteira_iu2 = [1.0; ganho_cumulativo_carteira_iu2]
plot(ganho_cumulativo_carteira_iu2)









#Vari??ncia individual de cada a????o.
?? = zeros(length(ACOES))
auxiliar = zeros(length(ACOES))
for i in 1:length(R)


    for j in 1:length(D[:,1])
        auxiliar[i] += (D[j,i] - R[i])^2

    end
    ??[i] = auxiliar[i]/length(D[:,1])
    
end

#Pode ser calculada tamb??m da seguinte maneira (bem mais compacta)
Var = diag(Q)
Desvio_padr??o = sqrt.(Var)