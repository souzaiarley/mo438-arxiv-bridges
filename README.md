# Citações entre áreas e integração estrutural na rede de citações do arXiv

**Aluno:** Iarley Natã Lopes Souza\
**Disciplina:** MO438 — Redes Complexas\
**Professor:** Dr. Ruben Interian\
**Instituição:** Instituto de Computação — Unicamp

Projeto final da disciplina. Este repositório reúne a implementação, os resultados e os testes
das entregas F1 (caracterização da rede) e F2 (objetivos 2 a 7).

## Pergunta

As citações entre categorias distintas do arXiv (*interárea*) ocupam posições estruturais
diferentes das citações dentro de uma mesma categoria (*intra-área*)? Quanto contribuem para
manter a rede conectada?

## Dados

A rede é a [`ogbn-arxiv`](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv), do Open Graph
Benchmark (Hu et al., 2020), derivada do Microsoft Academic Graph:
* 169.343 artigos de Ciência da Computação, publicados entre 1971 e 2020;
* 1.166.243 citações, das quais 65,5% são intra-área;
* 40 categorias, uma por artigo.

A análise usa o grafo dirigido (quem cita quem) e o não dirigido (conectividade). As instâncias
GraphML estão em `outputs/graphml/`, versionadas por Git LFS. O arquivo original e os caches
ficam em `data/` e não são versionados.

## Análises

* Caracterização: graus, componentes, clusterização × transitividade, distâncias e diâmetro.
* Fluxos de citação entre categorias, com modelos nulos de volume e idade.
* Sobreposição de vizinhança e pontes das arestas intra e interárea.
* Remoção progressiva de arestas, com controles uniformes e pareados por grau.
* Comunidades (Leiden) comparadas às categorias.
* Ordenação aproximada das categorias pela direção das citações.

## Principais resultados

1. A rede é esparsa e conexa (distância média 5,75; diâmetro 25). A clusterização média
   (0,257) supera muito a transitividade (0,016), porque poucos artigos de grau alto concentram
   as triplas.
2. 24 das 40 categorias citam majoritariamente outras áreas. Em 34 dos 58 pares de alto volume,
   a direção dos fluxos não é explicada pelo volume e pela idade dos artigos (após a correção
   de Holm).
3. Citações interárea compartilham menos vizinhos (0,046 contra 0,072) e são 1,6 vez mais
   frequentemente pontes locais.
4. Removê-las reduz mais a maior componente do que remover ao acaso a mesma quantidade: com 30%
   das arestas retiradas, ficam 87,13% dos artigos, contra 92,65%.
5. O grau dos extremos explica 64–67% da diferença inicial. Depois, 60–64% dela vem de artigos
   que ficam isolados.
6. As comunidades são mais modulares que as categorias (Q = 0,712 contra 0,495), mas só
   moderadamente parecidas com elas (NMI = 0,41).
7. A melhor ordem das categorias respeita 58,2% do peso das citações interárea, contra 50% ao
   acaso, com áreas de fundamentos antes das aplicadas.

**Em resumo:** as citações interárea têm papel estrutural diferenciado, com menos redundância
local, mais pontes e maior efeito sobre a conectividade. Ainda assim, a rede mantém muitas
ligações alternativas entre áreas. Os resultados são associações em um grafo estático, não
efeitos causais.

## Estrutura do repositório

```text
.
├── config/default.yaml         # configuração: caminhos, parâmetros e sementes
├── data/                       # arquivo original e caches (não versionado)
├── outputs/                    # resultados (CSV, JSON e figuras)
│   ├── graphml/                # instâncias da rede (Git LFS)
│   ├── obj1_characterization/  # caracterização da rede
│   ├── obj1_clustering/        # clusterização × transitividade
│   ├── edge_classes/           # classificação intra/interárea das arestas
│   ├── obj2_flows/             # fluxos entre categorias
│   ├── obj3_overlap/           # sobreposição de vizinhança e pontes
│   ├── obj4_5_removal/         # remoção progressiva e controles
│   ├── obj6_communities/       # comunidades
│   ├── obj7_ordering/          # ordenação das categorias
│   ├── integration/            # análise cruzada entre objetivos
│   └── review/                 # verificações cruzadas
├── src/mo438/                  # código (comando `mo438`)
├── tests/                      # testes (pytest)
└── pyproject.toml, uv.lock     # ambiente (uv; Python 3.12)
```

## Reprodução

Requer [uv](https://docs.astral.sh/uv/) e [Git LFS](https://git-lfs.com/):

```bash
git lfs pull && uv sync --frozen
uv run --frozen mo438 --help   # um subcomando por análise
uv run --frozen pytest
```
