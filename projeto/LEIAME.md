# Separação de fotos por rosto — solenidade

Tudo roda **local**. Nenhuma imagem e nenhum embedding sai da máquina.

## Instalação (~5 min)

**Não use `python3` puro.** Aqui ele resolve para 3.9.6 (CommandLineTools, via
pyenv `system`), e os scripts não rodam nessa versão: `Path.hardlink_to()` pede
3.10+, `HDBSCAN` pede scikit-learn ≥ 1.3, e o `onnxruntime` nem publica wheel
para cp39. Use o 3.12 explicitamente:

```bash
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install insightface onnxruntime opencv-python-headless pillow pillow-heif scikit-learn numpy
```

No primeiro `indexar.py` ele baixa o modelo `buffalo_l` (~300 MB). Precisa de
internet **uma vez**.

> Se você interromper o `venv` com Ctrl+C, ele deixa uma `.venv` pela metade —
> com os symlinks do python mas **sem `activate` e sem `pip`**. Não tente
> consertar: `rm -rf .venv` e comece de novo.

> **CoreML está quebrado nesta máquina.** O provider carrega, e aí falha a cada
> imagem na inferência. Como o `indexar.py` engole o erro por foto e segue, o
> sintoma é traiçoeiro: `erro_det` em 100% das fotos, nenhuma linha de
> progresso, código de saída 0 e um alegre `Pronto. 0 rostos`. Por isso o
> padrão é **CPU**; `--coreml` opta por voltar. Na CPU dá ~0,8 foto/s.

## O comando do dia a dia: `alvo`

Para achar uma pessoa e montar a pasta dela, é só isto:

```bash
alvo beltrano               # analisa, escolhe o corte sozinho e monta a pasta
alvo Sicrano --so-analise   # só o histograma, não cria nada
alvo -help                  # todas as opções
```

O argumento é um **pedaço do nome** do arquivo de referência em `Alvos/`, sem
extensão e sem ligar para maiúscula: `beltrano` acha `Alvos/prf-beltrano.png`.
Se casar com mais de um arquivo, ele lista e para.

> Os nomes deste manual — Fulano, Beltrano, Sicrano, Ciclano, Deltrano, Zutano,
> Mengano — são fictícios. Os números ao lado deles são medições reais desta
> base. Os alvos de verdade estão em `Alvos/`, que o `.gitignore` não versiona.

**Alvo novo são dois passos:** largue a foto em `Alvos/` e rode. O nome do
arquivo vira o nome da pasta e o nome no cadastro, então nomeie o arquivo como
você quer a pasta — `Alvos/Maj-Fulano.jpeg` → `Alvos/Maj-Fulano/`.

Não precisa reindexar. O `indexar.py` descreve o *evento*, não as pessoas: os
rostos já estão medidos. Alvo novo é só um vetor comparado contra eles, uns 10
segundos, quase tudo carregando o modelo.

O alias mora no `~/.zshrc` e aponta para o python da `.venv` de propósito,
para não depender de a venv estar ativada nem de qual `python3` o pyenv serve:

```zsh
alias alvo='/Users/vicente/Ambiente/dintel/fotos/.venv/bin/python /Users/vicente/Ambiente/dintel/fotos/projeto/alvo.py'
```

### O que ele faz que o `buscar.py` sozinho não faz

1. Acha a referência em `Alvos/` por pedaço do nome.
2. Confirma que o detector achou rosto nela, descendo o `det_size`
   1024 → 640 → 480 → 320. Recorte justo demais devolve **zero rostos sem erro
   nenhum** — é a falha silenciosa do projeto.
3. Escolhe o `--limiar` no maior degrau do histograma de distâncias, em vez do
   padrão 0.45.
4. Pergunta ao cadastro se algum rosto escolhido está **mais perto de outra
   pessoa** já registrada.
5. Chama o `buscar.py` com o limiar escolhido. Quem escreve a pasta é sempre
   ele; o `alvo.py` não cria nem apaga foto.

Por padrão ele fica calado. O `--verbose` traz de volta o barulho da
insightface carregando os 5 modelos do `buffalo_l` — útil só para diagnosticar
carga de modelo ou queda de `det_size`.

### Opções

`alvo -help` (ou `-h`, ou `--help`) imprime tudo isto, mais os 5 passos e o
guia de leitura da saída.

**Corte**

| opção | o que faz |
|---|---|
| `--limiar D` | força o corte em vez de achar o degrau. Mesmo sentido do `--limiar` do `buscar.py`: menor = mais conservador |
| `--so-analise` | mostra o histograma e para. Não cria pasta nenhuma |
| `--mostrar N` | quantos rostos listar (padrão 45; `0` desliga a lista) |
| `--verbose` | o barulho da insightface ao carregar os modelos. `--verboso` é sinônimo |

**Entrega**

| opção | o que faz |
|---|---|
| `--copiar` | copia de verdade em vez de hardlink. Só para pendrive ou zip |
| `--realimentar N` | guarda N rostos encontrados no cadastro, por diversidade de ângulo. Só depois de conferir com o olho |
| `--nao-memorizar` | não grava a referência no cadastro |

**Caminhos** — todos derivados da posição do script, então o comando roda de
qualquer diretório.

| opção | padrão |
|---|---|
| `--alvos DIR` | `Alvos/` — pasta das referências **e** das entregas |
| `--db ARQ` | `solenidade.db` — índice do evento, feito pelo `indexar.py` |
| `--memoria ARQ` | `pessoas.db` — cadastro de rostos entre eventos |

### Lendo a saída

| coluna | o que é |
|---|---|
| `d(busca)` | distância contra a referência + o cadastro. É esta que vira `--limiar` |
| `d(núcleo)` | a mesma, com os acertos óbvios (≤ 0.35) somados como vetor extra. Separa perfil e cabeça baixa do fundo |
| `x` | rosto fora do corte |
| `area%` | tamanho do rosto no quadro. Abaixo de ~0.3% é gente de multidão: o vetor tende ao "rosto médio" e casa fraco com todo mundo |

Ele imprime os **três** maiores degraus de propósito. Quando os dois primeiros
estão colados, a escolha do corte volta a ser sua — no `prf-beltrano` foi 0.080
contra 0.078, e o vice daria 32 fotos em vez de 33.

### Antes do degrau, olhe o `mais perto`

Esta é a checagem que evita entregar lixo. Distância do melhor acerto de cada
alvo, medida só com o vetor da foto de referência:

| alvo | mais perto | rostos ≤ 0.35 |
|---|---|---|
| Prof-Zutano | 0.012 | 51 |
| Cel-Ciclano | 0.016 | 277 |
| Dr-Sicrano | 0.164 | 17 |
| prf-beltrano | 0.246 | 6 |
| **CEL_MENGANO** | **0.489** | **0** |

Alvo que está no evento aparece com melhor acerto entre 0.01 e 0.25 e um
punhado de rostos no núcleo. O `CEL_MENGANO` veio ao dobro da distância do
pior caso, com núcleo vazio — e mesmo assim o detector de degrau achou o vão
**mais largo já visto** (0.108) e selecionou 31 fotos.

Eram `IMG_0630` a `IMG_0660`: uma rajada contígua, sem buraco, todas com área
~0.5% e nitidez ~115. **Gap largo prova separação, não identidade.** Ali o vão
separava *uma sequência de fotos* do resto, não *a pessoa* do resto.

Sintomas de referência ruim ou pessoa ausente:

- `mais perto` acima de ~0.35 e o aviso `! nenhum rosto abaixo de 0.35`;
- conjunto aceito que é uma faixa contígua de nomes de arquivo;
- área e nitidez quase idênticas em todos — é um rosto seguido quadro a quadro.

O conserto costuma ser a referência, não o limiar: **domínio importa mais que
resolução.** Um frame de webcam do Deltrano parou em 0.375 sem gap nenhum; um
recorte de 185×192 tirado do próprio evento foi a 0.039 com gap limpo.

## A pipeline completa

Só é necessária uma vez por evento — depois disso, o dia a dia é o `alvo`.

```bash
# 1. INDEXAR — a etapa cara. ~40 min para as 1837 fotos na CPU (~0,8 foto/s).
#    Pode interromper e rodar de novo: ele retoma de onde parou.
python indexar.py ../Fotos --db solenidade.db

# 2. BUSCAR por referência — o que o `alvo` automatiza
python buscar.py ../Alvos/Cel-Ciclano.jpeg --db solenidade.db --saida ../Alvos

# 3. AGRUPAR todo mundo, inclusive quem você não sabe quem é
python agrupar.py --db solenidade.db

# 4. Abra ./clusters no Finder (modo galeria) e renomeie o que reconhecer:
#       cluster_007.jpg  ->  cluster_007 - Cap Silva.jpg

# 5. DISTRIBUIR
python distribuir.py --db solenidade.db --saida ../Entrega
python distribuir.py --db solenidade.db --saida ../Entrega --copiar   # pendrive/zip
```

> `indexar.py` é idempotente: re-rodar depois de um crash não custa nada.
> `agrupar.py` **não é** — ele faz `DROP TABLE grupos` a cada execução, o que
> descarta a numeração que os seus renomeios do passo 4 usam. Reagrupar
> significa renomear de novo.

> `buscar.py` e `distribuir.py` têm o mesmo `--saida saida` padrão. Aponte um
> deles para outro lugar, ou você mistura arquivos com prefixo de distância
> com arquivos sem.

## Cadastro de rostos (pessoas.db)

Toda busca por referência **grava os vetores no cadastro** `pessoas.db`, sob o
nome do arquivo de referência. A partir daí, achar a mesma pessoa numa pasta de
fotos nova não precisa mais de foto de referência nenhuma:

```bash
python indexar.py ~/Fotos/OutroEvento --db outro.db     # indexa o evento novo
python buscar.py --conhecidos --db outro.db --saida entrega --copiar
python buscar.py --conhecidos --quem "Cel-Ciclano" --db outro.db   # só uma pessoa
```

`--conhecidos` nem carrega o modelo de reconhecimento — é uma multiplicação de
matriz contra o índice. Roda em fração de segundo.

```bash
python pessoas.py                                  # quem está cadastrado
python pessoas.py --renomear "Cel-Ciclano" "Cel Ciclano"
python pessoas.py --remover "Fulano"
```

O cadastro casa nome por **string exata**. Por isso uma segunda foto do mesmo
alvo não pode entrar como `prf-beltrano-2.png`: viraria uma pessoa separada, com
pasta separada, e você perderia justamente o ganho de ter vários vetores sob o
mesmo nome. Para somar ângulos a alguém que já existe, use `--realimentar`.

### Realimentar (importante para o recall)

Uma pessoa pode ter **vários** vetores, e a busca usa a menor distância contra
qualquer um deles. Isso resolve o problema de perfil: o ArcFace é treinado em
rostos frontais, então uma referência frontal única perde as fotos de lado.

Depois de conferir um resultado, devolva os rostos encontrados ao cadastro:

```bash
alvo ciclano --realimentar 12
```

Ele escolhe 12 rostos *espalhados* (não 12 frontais quase iguais), cobrindo
ângulos. Medido nesta base: com 1 vetor frontal a busca achou 94 fotos; com 13
vetores achou 99, sem nenhum falso positivo — e as 5 novas eram perfis extremos
e cabeça baixa.

**Realimentar muda o limiar.** A comparação passa a ser a menor distância
contra N vetores, então TODAS as distâncias encolhem — inclusive as de outras
pessoas. O limiar que estava certo antes fica permissivo demais depois:

| | limiar seguro antes | depois de realimentar |
|---|---|---|
| Cel Ciclano (99 fotos) | 0.60 | 0.60 (separação era enorme) |
| Prom Deltrano (18 fotos)  | 0.65 | **0.55** |

Com o `alvo` isso se resolve sozinho — ele mede no mesmo espaço que o
`buscar.py` usa, cadastro incluído, e re-deriva o corte na rodada seguinte.

E o ganho não é garantido: no Ciclano realimentar rendeu +5 fotos (perfis que a
referência frontal perdia); no Deltrano rendeu zero, porque as 12 fotos já cobriam
os ângulos.

Rostos borrados são descartados automaticamente (`--realimentar-nitidez`,
padrão 60): um rosto sem foco gera um vetor que puxa para o "rosto médio" e
passa a casar com outros borrados. Nesta base a mediana de nitidez é 149 e só
10% dos rostos ficam abaixo de 60, então o corte é permissivo — o que ele barra
é quadro fora de foco de rajada.

> Só realimente **depois** de conferir com o olho. Realimentar um resultado com
> gente errada contamina o cadastro e degrada em silêncio toda busca futura.
> `pessoas.py --esquecer-matches NOME` desfaz, preservando as referências
> originais.

## Ajuste fino

| Sintoma | O que fazer |
|---|---|
| Poucos rostos detectados em fotos de formação | `indexar.py --det-size 1600` (reindexar, ~2x mais lento) |
| Mesma pessoa em vários clusters | `agrupar.py --limiar 1.05` |
| Pessoas diferentes no mesmo cluster | `agrupar.py --limiar 0.75` |
| Busca perdendo fotos | `alvo NOME --limiar 0.55`, ou `--realimentar 12` |
| Busca perdendo fotos de perfil | `--realimentar` resolve melhor que subir o limiar |
| Busca trazendo gente errada | `alvo NOME --limiar 0.35` |
| Referência não detecta rosto nenhum | recorte menos justo, ou uma foto do próprio evento |
| Muito rosto borrado/minúsculo | `agrupar.py --nitidez-min 60 --area-min 0.002` |
| Entregar só fotos onde a pessoa é o assunto | `distribuir.py --area-min 0.01` |

## Regras

- `Fotos/` é **somente leitura**. Nada é movido, renomeado ou apagado. Nunca.
- Saída usa **hardlink** — 50 pessoas × 100 fotos custa ~0 bytes. O que o `du`
  mostra é o mesmo dado contado duas vezes, não espaço novo. Só use `--copiar`
  no momento da entrega (pendrive, zip).
- O prefixo numérico no nome do arquivo é a distância: os primeiros arquivos da
  pasta são os mais confiáveis. **Confira de baixo para cima**, que é onde mora
  o erro.
- **A decisão do `alvo` é 100% numérica.** Ele garante que o conjunto está
  separado do resto, não que seja a pessoa certa. Espere 70–85% de acerto,
  não 99%.

## LGPD

`solenidade.db` e `pessoas.db` contêm **dado biométrico** (art. 5º, II) de
militares — e `pessoas.db` ainda associa o vetor a um **nome**, o que o torna
mais sensível que o índice do evento.

Não versione, não suba pra nuvem, não compartilhe — nem como contexto para
ferramenta remota. Guarde offline ou apague depois da entrega — mas note que
ele é o que torna a próxima solenidade barata (as pessoas já ficam
reconhecidas). Se for guardar, vale alinhar com o comando antes.

> O `.gitignore` deste diretório cobre `Fotos/`, `Alvos/` e `.venv/`, mas
> **não** cobre `*.db`. Confira antes de qualquer `git add .`.
