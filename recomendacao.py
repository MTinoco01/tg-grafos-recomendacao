# Trabalho P2 - Teoria dos Grafos (UFGD / FACET)
# Sistema de Recomendação de Filmes/Séries com TF-IDF, K-Means e Grafos
# Segue o roteiro do professor, passo a passo.
#
# Uso:
#   py recomendacao.py                      -> pede o nome do filme/série no terminal
#   py recomendacao.py "Iron Man" "Frozen"  -> roda direto para os títulos informados

# %% Passo 1: Importação das bibliotecas
import os
import sys
import math
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # salva as figuras em PNG em vez de abrir janela
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore", message="Glyph .* missing from font")  # nomes em japonês etc.

PASTA = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(PASTA, "disney_plus_titles.csv")
SAIDA = os.path.join(PASTA, "saida")


# %% Passo 2: Leitura e pré-processamento do dataset
print("=" * 70)
print("PASSO 2 - Leitura e pré-processamento")
print("=" * 70)

dados = pd.read_csv(caminho)
print(dados.head())

print("\nColunas:", list(dados.columns))
print("Formato:", dados.shape)
print("Descrições faltando:", dados["description"].isna().sum())
print("Exemplo:", dados["description"].iloc[0])


def preprocessar_lista(texto):
    """Converte 'A, B, C' em ['A', 'B', 'C']. Valores nulos viram lista vazia."""
    if pd.isna(texto):
        return []
    return [item.strip() for item in str(texto).split(",") if item.strip()]


print("\nTeste:", preprocessar_lista("Mark Hamill, Harrison Ford"), preprocessar_lista(np.nan))

for coluna in ["cast", "director", "listed_in", "country"]:
    dados[coluna] = dados[coluna].apply(preprocessar_lista)

dados["description"] = dados["description"].fillna("")
print(dados[["title", "cast", "director", "listed_in", "country"]].head(3))


# %% Passo 3: Transformação do texto em vetores TF-IDF
print("\n" + "=" * 70)
print("PASSO 3 - TF-IDF")
print("=" * 70)

vetorizador = TfidfVectorizer(
    stop_words="english",           # descrições estão em inglês
    max_features=2000,              # vocabulário de no máximo 2000 palavras
    token_pattern=r"[A-Za-z]{2,}",  # só palavras com letras e pelo menos 2 caracteres
    min_df=2,                       # ignora palavras que aparecem em menos de 2 filmes
    max_df=0.8,                     # ignora palavras presentes em mais de 80% dos filmes
)
matriz_tfidf = vetorizador.fit_transform(dados["description"])

print("Formato da matriz TF-IDF:", matriz_tfidf.shape)
print(" • Número de filmes:", matriz_tfidf.shape[0])
print(" • Tamanho do vocabulário:", matriz_tfidf.shape[1])
print(" • 20 primeiras palavras:", list(vetorizador.get_feature_names_out()[:20]))
total = matriz_tfidf.shape[0] * matriz_tfidf.shape[1]
print(f" • Valores diferentes de zero: {matriz_tfidf.nnz} de {total} "
      f"({100 * matriz_tfidf.nnz / total:.2f}%)")

# 3.1 Similaridade por cosseno
matriz_similaridade = cosine_similarity(matriz_tfidf)
print("\nFormato da matriz de similaridade:", matriz_similaridade.shape)

matriz_temp = matriz_similaridade.copy()
np.fill_diagonal(matriz_temp, -1)
maior_similaridade = matriz_temp.max()
i, j = np.unravel_index(matriz_temp.argmax(), matriz_temp.shape)
print(f"Maior similaridade entre filmes diferentes: {maior_similaridade:.3f}")
print(f" • {dados.iloc[i]['title']} × {dados.iloc[j]['title']}")


def buscar_titulo(nome_digitado):
    """Títulos que CONTÊM o texto (sem diferenciar maiúsculas).
    O título exato vem primeiro; depois o que termina com o texto e o mais curto."""
    n = nome_digitado.lower()
    encontrados = dados.loc[dados["title"].str.contains(nome_digitado, case=False,
                                                         na=False, regex=False), "title"].tolist()
    encontrados.sort(key=lambda t: (t.lower() != n, not t.lower().endswith(n), len(t)))
    return encontrados


def similares_cosseno(indice, k=5):
    """Lista (indice, similaridade, titulo) dos k mais parecidos pelo texto."""
    similaridades = matriz_similaridade[indice]
    filmes_similares = []
    for idx, sim in enumerate(similaridades):
        if idx != indice and sim > 0:
            filmes_similares.append((idx, sim, dados.loc[idx, "title"]))
    filmes_similares.sort(key=lambda x: x[1], reverse=True)
    return filmes_similares[:k]


def testar_filme_especifico(nome_filme, k=5):
    """Mostra os k filmes com maior similaridade de cosseno (somente texto, sem grafo)."""
    encontrados = buscar_titulo(nome_filme)
    if not encontrados:
        print(f"Filme '{nome_filme}' não encontrado")
        return
    indice = dados.index[dados["title"] == encontrados[0]][0]
    print(f"\n# TESTANDO: {dados.loc[indice, 'title']}")
    print(f"- Descrição: {dados.loc[indice, 'description']}")
    print("\nFILMES SIMILARES (cosseno):")
    for pos, (idx, sim, tit) in enumerate(similares_cosseno(indice, k), 1):
        print(f"  {pos}. {tit} → {sim:.3f}")


testar_filme_especifico("Star Wars")


# %% Passo 4: Agrupamento com MiniBatchKMeans
print("\n" + "=" * 70)
print("PASSO 4 - MiniBatchKMeans")
print("=" * 70)

kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
clusters = kmeans.fit_predict(matriz_tfidf)
dados["cluster"] = clusters

print("10 maiores clusters (cluster: quantidade de filmes):")
print(dados["cluster"].value_counts().head(10).to_string())

# Explorando um cluster: o cluster do filme Iron Man
numero_cluster = int(dados.loc[dados["title"] == "Marvel Studios' Iron Man", "cluster"].iloc[0])
print(f"\nTítulos do cluster {numero_cluster}:")
print(dados.loc[dados["cluster"] == numero_cluster, "title"].head(15).tolist())


# %% Passo 5: Construção do grafo com NetworkX
print("\n" + "=" * 70)
print("PASSO 5 - Grafo")
print("=" * 70)

similaridade_minima = 0.15  # limiar para ligar dois filmes por similaridade de texto


def criar_grafo(dados, matriz_similaridade):
    print("Montando o grafo...")
    G = nx.Graph()

    # 5.1 Vértices de filmes
    for idx, filme in dados.iterrows():
        G.add_node(f"filme_{idx}", tipo="filme", titulo=filme["title"], cluster=int(filme["cluster"]))

    # 5.2 Arestas filme ↔ filme por similaridade
    n = len(dados)
    for i in range(n):
        for j in range(i + 1, n):
            if matriz_similaridade[i, j] > similaridade_minima:
                G.add_edge(f"filme_{i}", f"filme_{j}", tipo="similaridade",
                           peso=float(matriz_similaridade[i, j]))

    # 5.3 a 5.6 Atores, diretores, países e categorias
    ligacoes = [
        ("cast", "ator", "elenco"),
        ("director", "diretor", "direcao"),
        ("country", "pais", "origem"),
        ("listed_in", "categoria", "genero"),
    ]
    for coluna, tipo_no, tipo_aresta in ligacoes:
        for idx, filme in dados.iterrows():
            for nome in filme[coluna]:
                if nome:
                    no_id = f"{tipo_no}_{nome}"
                    G.add_node(no_id, tipo=tipo_no, nome=nome)
                    G.add_edge(f"filme_{idx}", no_id, tipo=tipo_aresta)

    print("\nESTATÍSTICAS DO GRAFO")
    print(" • Vértices:", G.number_of_nodes())
    print(" • Arestas:", G.number_of_edges())
    return G


grafo = criar_grafo(dados, matriz_similaridade)

# Análise estrutural do grafo
print("\nVértices por tipo:", dict(Counter(nx.get_node_attributes(grafo, "tipo").values())))
print("Arestas por tipo:", dict(Counter(nx.get_edge_attributes(grafo, "tipo").values())))

atores = [(grafo.nodes[n]["nome"], g) for n, g in grafo.degree() if grafo.nodes[n]["tipo"] == "ator"]
atores.sort(key=lambda x: x[1], reverse=True)
print("\n10 atores com maior grau (mais filmes no catálogo):")
for nome, grau in atores[:10]:
    print(f"  {nome}: {grau}")

componentes = list(nx.connected_components(grafo))
print("\nO grafo é conexo?", nx.is_connected(grafo))
print("Componentes conexas:", len(componentes))
print("Tamanho da maior componente:", max(len(c) for c in componentes), "vértices")


# %% Passo 6: Desenho e visualização do subgrafo
CORES = {"filme": "#9ecae1", "ator": "#c994c7", "diretor": "orange",
         "categoria": "lightgreen", "pais": "yellow"}


def encontrar_no_filme(grafo, titulo):
    """Retorna o identificador do vértice cujo atributo 'titulo' é igual ao informado."""
    for no, atributos in grafo.nodes(data=True):
        if atributos["tipo"] == "filme" and atributos["titulo"] == titulo:
            return no
    return None


def visualizar_grafo(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return

    filmes_similares = []
    for vizinho in grafo.neighbors(filme_node):
        if grafo.nodes[vizinho]["tipo"] == "filme":
            filmes_similares.append((vizinho, grafo[filme_node][vizinho]["peso"],
                                     grafo.nodes[vizinho]["titulo"]))
    filmes_similares.sort(key=lambda x: x[1], reverse=True)
    top_k = filmes_similares[:k]

    nos = {filme_node}
    nos.update(v for v in grafo.neighbors(filme_node) if grafo.nodes[v]["tipo"] != "filme")
    for vizinho, _, _ in top_k:
        nos.add(vizinho)
        nos.update(v for v in grafo.neighbors(vizinho) if grafo.nodes[v]["tipo"] != "filme")
    sub = grafo.subgraph(nos)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(sub, k=1, iterations=50, seed=42)
    cores = [CORES.get(sub.nodes[n]["tipo"], "gray") for n in sub.nodes()]
    rotulos = {}
    for n in sub.nodes():
        if sub.nodes[n]["tipo"] == "filme":
            rotulos[n] = sub.nodes[n]["titulo"][:40]
        else:
            rotulos[n] = sub.nodes[n]["nome"]
    nx.draw_networkx_nodes(sub, pos, node_color=cores, node_size=400, edgecolors="black")
    nx.draw_networkx_edges(sub, pos, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(sub, pos, rotulos, font_size=9)
    for tipo, cor in CORES.items():
        plt.scatter([], [], c=cor, edgecolors="black", label=tipo)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.title(f"Subgrafo de '{titulo_filme}' e seus {k} filmes mais similares", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(SAIDA, exist_ok=True)
    nome_arq = "".join(ch if ch.isalnum() else "_" for ch in titulo_filme)[:50]
    arquivo = os.path.join(SAIDA, f"subgrafo_{nome_arq}.png")
    plt.savefig(arquivo, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSubgrafo salvo em: {arquivo}")


# %% Passo 7: Recomendação com a métrica de Adamic-Adar
def recomendar_adamic_adar(grafo, titulo_filme, k=5):
    """Retorna os k filmes com maior índice de Adamic-Adar em relação ao filme informado."""
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return []
    candidatos = [n for n, t in grafo.nodes(data="tipo") if t == "filme" and n != filme_node]
    pares = [(filme_node, c) for c in candidatos]
    pontuacoes = nx.adamic_adar_index(grafo, pares)
    resultado = [(grafo.nodes[v]["titulo"], p) for _, v, p in pontuacoes if p > 0]
    resultado.sort(key=lambda x: x[1], reverse=True)
    return resultado[:k]


def adamic_adar_manual(grafo, u, v):
    """DESAFIO: Adamic-Adar calculado "na mão"."""
    return sum(1 / math.log(grafo.degree(z)) for z in nx.common_neighbors(grafo, u, v))


def conferir_desafio(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    candidatos = [n for n, t in grafo.nodes(data="tipo") if t == "filme" and n != filme_node]
    pares = [(filme_node, c) for c in candidatos]
    diferenca = max(abs(p - adamic_adar_manual(grafo, u, v))
                    for u, v, p in nx.adamic_adar_index(grafo, pares))
    print(f"\n[Desafio] Maior diferença entre o Adamic-Adar manual e o do NetworkX: {diferenca:.2e}")


# %% Passo 8: Sistema de recomendação final
def recomendar(nome_digitado, k=5):
    print("\n" + "=" * 70)
    print(f"RECOMENDAÇÃO PARA: '{nome_digitado}'")
    print("=" * 70)
    encontrados = buscar_titulo(nome_digitado)
    if not encontrados:
        print(f"Nenhum título encontrado para '{nome_digitado}'.")
        return
    titulo = encontrados[0]
    if len(encontrados) > 1:
        print(f"{len(encontrados)} títulos encontrados. Usando: {titulo}")
        print("Outras opções:", encontrados[1:4])
    else:
        print("Título:", titulo)

    indice = dados.index[dados["title"] == titulo][0]
    print("\n[1] Similaridade de cosseno (descrição):")
    for pos, (_, sim, tit) in enumerate(similares_cosseno(indice, k), 1):
        print(f"  {pos}. {tit} → {sim:.3f}")

    print("\n[2] Adamic-Adar (grafo):")
    for pos, (tit, pont) in enumerate(recomendar_adamic_adar(grafo, titulo, k), 1):
        print(f"  {pos}. {tit} → {pont:.3f}")

    conferir_desafio(grafo, titulo, k)
    visualizar_grafo(grafo, titulo, k)


if __name__ == "__main__":
    nomes = sys.argv[1:] or [input("\nDigite o nome do filme/série: ").strip()]
    for nome in nomes:
        recomendar(nome)
