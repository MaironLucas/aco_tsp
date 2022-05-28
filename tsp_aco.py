import math
import random
from scipy.spatial import distance
from pprint import pprint
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def initMatriz(dic_cities, feromonio_inicial):
    num_cities = len(dic_cities)
    paths = np.zeros(
        (num_cities, num_cities, 2))
    for i in range(len(dic_cities)):
        for j in range(i):
            if i != j:
                peso = distance.euclidean([dic_cities[i]['cordX'], dic_cities[i]['cordY']], [
                    dic_cities[j]['cordX'], dic_cities[j]['cordY']])
                paths[i][j] = np.array([peso, feromonio_inicial])
                paths[j][i] = np.array([peso, feromonio_inicial])
            else:
                paths[i][j] = None
    return paths


@jit(nopython=True)
def somatorioPesoFeromonio(cidades_nao_visitadas, cidade_atual, paths, alpha, beta):
    somatorio = 0.0
    for cidade in cidades_nao_visitadas:
        feromonio = paths[cidade_atual][cidade][1]
        distancia = paths[cidade_atual][cidade][0]
        somatorio += (math.pow(feromonio, alpha)
                      * math.pow(1.0 / distancia, beta))
    return somatorio


def obterCusto(rota_formiga, paths, num_cidades):
    custo = 0.0
    for i in range(num_cidades - 1):
        origem = rota_formiga[i]
        destino = rota_formiga[i+1]
        custo += paths[origem][destino][0]
    custo += paths[rota_formiga[0]][rota_formiga[num_cidades - 1]][0]
    return custo


def mapearRotas(cidades_visitadas, paths, melhor_rota, menor_custo, num_cidades, num_formigas):
    custos_formigas = []
    for k in range(num_formigas):
        custo = obterCusto(
            cidades_visitadas[k], paths, num_cidades)
        if not menor_custo:
            menor_custo = custo
            melhor_rota = cidades_visitadas[k]
        else:
            if custo < menor_custo:
                menor_custo = custo
                melhor_rota = cidades_visitadas[k]
        custos_formigas.append(custo)
    return custos_formigas, menor_custo, melhor_rota


def verificaArestaFormiga(origem, destino, rota_formiga):
    for i in range(len(rota_formiga) - 1):
        if ((rota_formiga[i] == origem and rota_formiga[i+1] == destino) or (rota_formiga[i] == destino and rota_formiga[i+1] == origem)):
            return True
    if (rota_formiga[0] == origem and rota_formiga[len(rota_formiga) - 1] == destino) or (rota_formiga[0] == destino and rota_formiga[len(rota_formiga) - 1] == origem):
        return True
    return False


def sobreescreverFeromonio(num_cidades, cidades_visitadas, custos_formigas, paths):
    for origem in tqdm(range(num_cidades)):
        for destino in range(origem):
            somatorio_feromonio = 0.0
            for k in range(num_formigas):
                if verificaArestaFormiga(origem, destino, cidades_visitadas[k]):
                    somatorio_feromonio += 1 / custos_formigas[k]
            novo_feromonio = (1.0 - evaporacao) * \
                paths[origem][destino][1] + somatorio_feromonio
            paths[origem][destino][1] = novo_feromonio
            paths[destino][origem][1] = novo_feromonio
    return paths


def rodar(paths, num_formigas, iteracoes, alpha, beta, evaporacao, num_cidades, mapa):
    formigas_rotas = np.zeros((num_formigas, num_cidades))
    melhor_rota = np.zeros(num_cidades)
    menor_custo = None
    custos_iteracao = []
    for it in tqdm(range(iteracoes)):
        cidades_visitadas = []
        lista_cidades = [cidade for cidade in range(num_cidades)]
        for k in range(num_formigas):
            cidade_formiga = lista_cidades[random.randint(
                0, len(lista_cidades) - 1)]
            lista_cidades.remove(cidade_formiga)
            cidades_visitadas.append([cidade_formiga])
            formigas_rotas[k][0] = cidade_formiga
        for k in range(num_formigas):
            cidade_atual = int(formigas_rotas[k][0])
            for i in range(1, num_cidades):
                vizinhos = [cidade for cidade in range(num_cidades)]
                vizinhos.remove(cidade_atual)
                cidades_nao_visitadas = list(
                    set(vizinhos) - set(cidades_visitadas[k]))

                somatorio = somatorioPesoFeromonio(
                    cidades_nao_visitadas, cidade_atual, paths, alpha, beta)

                probabilidades = []
                intervalo = random.random()
                for j in range(len(cidades_nao_visitadas)):
                    cidade = cidades_nao_visitadas[j]
                    feromonio = paths[cidade_atual][cidade][1]
                    distancia = paths[cidade_atual][cidade][0]
                    prob = (math.pow(feromonio, alpha) * math.pow(
                        1.0 / distancia, beta)) / somatorio
                    if j == 0:
                        probabilidades.append(prob)
                    else:
                        probabilidades.append(prob + probabilidades[j-1])
                    if intervalo <= probabilidades[j]:
                        cidade_escolhida = cidades_nao_visitadas[j]
                        break
                cidades_visitadas[k].append(cidade_escolhida)
                formigas_rotas[k][i] = cidade_escolhida
                cidade_atual = cidade_escolhida

        custos_formigas, menor_custo, melhor_rota = mapearRotas(
            cidades_visitadas, paths, melhor_rota, menor_custo, num_cidades, num_formigas)
        custos_iteracao.append(menor_custo)
        for origem in tqdm(range(num_cidades)):
            for destino in range(origem):
                somatorio_feromonio = 0.0
                for k in range(num_formigas):
                    if verificaArestaFormiga(origem, destino, cidades_visitadas[k]):
                        somatorio_feromonio += 1 / custos_formigas[k]
                novo_feromonio = (1.0 - evaporacao) * \
                    paths[origem][destino][1] + somatorio_feromonio
                paths[origem][destino][1] = novo_feromonio
                paths[destino][origem][1] = novo_feromonio
    with open('output/' + mapa + '/custo_iteracao.txt', 'a') as arquivo:
        arquivo.write(str(custos_iteracao) + '\n')
        arquivo.close()
    return melhor_rota


def plotarSolucao(dic_cidades, solucao, cidade, num_solucao):
    xCord, yCord = [], []
    count = 0
    plt.clf()
    for i in solucao:
        xCord.append(dic_cidades[i]['cordX'])
        yCord.append(dic_cidades[i]['cordY'])
        if count == 0:
            primeiroPontoX = dic_cidades[i]['cordX']
            primeiroPontoY = dic_cidades[i]['cordY']
        count += 1
    xCord.append(primeiroPontoX)
    yCord.append(primeiroPontoY)
    plt.plot(xCord, yCord)
    path = 'output/' + str(cidade) + '/' + str(num_solucao) + '.png'
    plt.savefig(path)


if __name__ == "__main__":
    raw = {}
    djibout = 'djibout'
    luxemburgo = 'Luxemburgo'
    oma = 'Omã'
    cidade = djibout
    with open('input/' + cidade + '.txt') as input_file:
        for line in input_file:
            city, cordX, cordY = (
                item.strip() for item in line.split(' '))
            raw[int(city)] = dict(
                zip(('cordX', 'cordY'), (float(cordX), float(cordY))))

    cities = {}
    count = 0

    for key, value in raw.items():
        if value not in cities.values():
            cities[count] = value
            count += 1

    num_formigas = 38
    alpha = 1.0
    beta = 5.0
    iteracoes = 20
    evaporacao = 0.5
    feromonio_inicial = 0.000001
    quant_solucoes = 4
    for i in range(quant_solucoes):
        inicio = time.time()
        paths = initMatriz(cities, feromonio_inicial)
        melhor_rota = rodar(paths=paths, num_formigas=num_formigas, num_cidades=len(cities),
                            iteracoes=iteracoes, alpha=alpha, beta=beta, evaporacao=evaporacao, mapa=cidade)
        fim = time.time()
        plotarSolucao(cities, melhor_rota, cidade, i)
        with open('output/' + cidade + '/tempo.txt', 'a') as arquivo:
            tempo = fim - inicio
            arquivo.write(str(i) + '->' + str(tempo) + '\n')
            arquivo.close()
