# ALVO — busca de rostos em massas de fotos

Você tem um acervo de milhares de fotos e uma pergunta simples: **em quais
delas esta pessoa aparece?**

Responder isso na mão é inviável — 1837 fotos com 6466 rostos são horas de
olho humano por pessoa procurada, e o cansaço faz errar. Este projeto responde
em segundos, a partir de **uma única foto de referência**, e devolve uma pasta
com as fotos encontradas ordenadas da mais confiável para a mais duvidosa.

Tudo roda **local**. Nenhuma imagem e nenhum embedding sai da máquina — é
requisito, não preferência. O único momento em que a internet é usada é o
download do modelo, uma vez.

> O primeiro acervo processado foi uma solenidade militar, e por isso o banco
> padrão se chama `solenidade.db`. Isso é só um nome de arquivo: a ferramenta
> não sabe nada sobre solenidades, e serve para qualquer massa de fotos —
> outro evento, um acervo de câmeras, um backup de anos.

## Como funciona

Reconhecimento facial aqui não é "comparar duas fotos". São três etapas
distintas, e entender a diferença entre elas é o que permite ajustar o
resultado quando ele vem errado.

```
  foto  ──►  DETECÇÃO  ──►  recorte do rosto  ──►  EMBEDDING  ──►  512 números
                                                                        │
  foto de referência ──► ... mesmo caminho ... ──► 512 números          │
                                                          │             │
                                                          └──► DISTÂNCIA ──► corte
```

**1. Detecção — "onde há rostos nesta imagem?"**

Um modelo chamado SCRFD varre a foto e devolve uma caixa em volta de cada
rosto, com uma nota de confiança. Ele não faz ideia de *quem* é: só sabe que
ali tem um rosto.

O detector compara a imagem contra **âncoras de escala** — tamanhos de rosto
que ele espera encontrar. Isso tem uma consequência que morde na prática: um
rosto grande demais em relação ao quadro não casa com âncora nenhuma e **não é
detectado**. É por isso que uma referência recortada colada no rosto pode
devolver zero rostos, e por que aumentar a resolução da imagem não resolve
(o tamanho *relativo* não muda) — o que resolve é diminuir o `det_size`.

**2. Embedding — "descreva este rosto em números"**

O recorte do rosto passa por um segundo modelo, que devolve um vetor de **512
números**. Esse vetor é a assinatura matemática daquele rosto. Rostos da mesma
pessoa produzem vetores parecidos; de pessoas diferentes, vetores distantes.

O modelo é treinado para que a *direção* do vetor carregue a identidade, então
todo vetor é normalizado para comprimento 1 antes de ser guardado. Duas
consequências práticas:

- a semelhança entre dois rostos vira um **produto escalar** — sem divisão,
  sem raiz quadrada. A distância cosseno é literalmente `1 - (a · b)`;
- distância euclidiana e distância cosseno passam a ordenar igual, o que
  permite ao agrupamento usar a árvore do HDBSCAN em vez de materializar uma
  matriz N×N de distâncias.

Guardar o vetor sem normalizar quebra as duas coisas em silêncio.

**3. Distância e corte — "perto o bastante para ser a mesma pessoa?"**

Aqui está o trabalho de verdade. A distância vai de `0.0` (idênticos) a `2.0`,
e **não existe um número universal que separe "mesma pessoa" de "pessoa
diferente"**. O valor certo muda com a qualidade da referência, com o ângulo
dos rostos e com o próprio acervo.

O erro clássico é confiar num limiar fixo. O `alvo` faz outra coisa: mede a
distância da referência contra **todos** os rostos indexados, ordena, e procura
o **maior degrau** — o vão onde a população muda de "provavelmente ela" para
"provavelmente outra pessoa". O corte vai ali.

## A tecnologia

| peça | o que é | papel aqui |
|---|---|---|
| [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` | pacote com 5 modelos ONNX (~300 MB) | fornece detector e reconhecedor |
| `det_10g.onnx` | SCRFD, detector de rostos | acha as caixas. Entrada de tamanho livre — daí o `--det-size` |
| `w600k_r50.onnx` | ResNet-50 treinada com perda ArcFace sobre o WebFace600K | gera o vetor de 512 dimensões. Entrada 112×112 |
| ONNX Runtime | motor de inferência | roda os modelos. **CPU** por padrão aqui |
| SQLite | banco em arquivo único | o índice, e a interface entre os scripts |
| NumPy | álgebra | a busca inteira é uma multiplicação de matriz |
| scikit-learn (HDBSCAN) | agrupamento por densidade | separa todo mundo sem referência nenhuma |
| Pillow + pillow-heif | leitura de imagem | inclusive HEIC do iPhone, e a correção de EXIF |
| OpenCV | visão computacional | variância do Laplaciano (nitidez) e as folhas de contato |

Dos 5 modelos do pacote, a pipeline usa **2**: `det_10g` e `w600k_r50`. Os
outros três (landmarks 2D, landmarks 3D, idade/gênero) são carregados junto
porque o `FaceAnalysis` carrega o pacote inteiro, e nunca são usados.

### Duas decisões que carregam o resultado

**A orientação EXIF é lida antes de tudo.** Uma foto em retrato lida sem
`ImageOps.exif_transpose` chega deitada, e o detector simplesmente não acha
rosto nenhum. Essa linha sozinha vale mais para a taxa de acerto do que
qualquer ajuste de limiar.

**As caixas são guardadas em coordenadas relativas (0..1).** A imagem é
reduzida para no máximo 2200px de lado antes da detecção; guardar a caixa em
pixels absolutos quebraria todo recorte feito depois.

## Arquitetura

Quatro scripts autônomos e um utilitário, rodados na ordem. Não é biblioteca
nem serviço — cada um faz uma etapa e conversa com o seguinte pelo **SQLite**.

```
  Fotos/            (somente leitura, nunca tocada)
    │
    ▼
  indexar.py   ──►  solenidade.db : fotos, rostos     [caro, ~40 min, retomável]
    │
    ├──► buscar.py    ──►  pasta por pessoa           [rápido, por referência]
    │      ▲
    │      └── pessoas.db : cadastro entre acervos
    │
    └──► agrupar.py   ──►  solenidade.db : grupos     [sem referência nenhuma]
           │                └──► clusters/*.jpg       (folhas de contato)
           │                       │
           │                       ▼  renomeio manual
           └──► distribuir.py ──►  pasta por pessoa
```

O `alvo.py` fica por cima do `buscar.py`: analisa o histograma, escolhe o
limiar e chama o `buscar.py` para escrever. Ele nunca cria nem apaga foto.

### Os dois bancos, e por que são separados

| banco | o que guarda | vida útil |
|---|---|---|
| `solenidade.db` | o índice de **um acervo**: cada foto, cada rosto, cada vetor | descartável — reindexar reconstrói |
| `pessoas.db` | o cadastro de **quem é quem**: nome → vetores | permanente, e é o que barateia o próximo acervo |

> **Nenhum dos dois vem no repositório, e nenhum dos dois deve entrar nele.**
> Os dois são criados **automaticamente** na primeira execução: o
> `solenidade.db` quando você roda o `indexar.py`, e o `pessoas.db` na primeira
> busca por referência. Se o arquivo não existir, o script cria o banco e as
> tabelas sozinho — não há passo de migração nem script de criação a rodar.
>
> Eles ficam de fora do git de propósito: contêm **dado biométrico** das
> pessoas do seu acervo (veja [LGPD](#lgpd)). O `.gitignore` cobre `*.db`,
> `*.db-wal` e `*.db-shm`.

Quem clona este repositório começa, portanto, **sem banco nenhum** — e é assim
que tem de ser. Do zero até a primeira busca:

```bash
python indexar.py /caminho/do/acervo --db meuacervo.db   # cria o indice
alvo Fulano --db meuacervo.db                            # cria o cadastro
```

O `indexar.py` é **retomável**: ele guarda o caminho de cada foto já processada
e pula o que já está lá, então interromper no meio e rodar de novo não custa
nada. É o que torna aceitável uma etapa de 40 minutos.

A separação é deliberada. O índice do acervo é grande e recriável; o cadastro
é pequeno e insubstituível — é o único arquivo do projeto que amarra um
**nome** a um **vetor biométrico**.

Com alguém já no cadastro, encontrá-lo num acervo novo dispensa foto de
referência e nem carrega o modelo de reconhecimento:

```bash
python indexar.py ~/Fotos/OutroAcervo --db outro.db
python buscar.py --conhecidos --db outro.db --saida entrega --copiar
```

### Esquema

| tabela | escrita por | colunas que importam |
|---|---|---|
| `fotos` | `indexar.py` | `caminho` (UNIQUE — é a chave de retomada), `n_rostos`, `status` |
| `rostos` | `indexar.py` | `emb` BLOB = float32[512] **já normalizado**, `x1..y2` relativos, `nitidez`, `area_rel` |
| `grupos` | `agrupar.py` | `rosto_id` → `grupo`. Recriada do zero a cada execução |
| `pessoas` / `referencias` | `memoria.py` | `nome`, e os vetores com `fonte` = `referencia` ou `match` |

Mudar esse esquema significa mexer em todos os consumidores.

### Qualidade: o que nunca entra no índice

Rosto ruim não vira erro de reconhecimento — vira poluição silenciosa. O
`indexar.py` descarta antes de gravar (constantes do módulo, não flags):

- `det_score` abaixo de **0.60** — o detector mesmo não confia;
- lado da caixa abaixo de **45 px** — não há informação ali.

Cada rosto aceito ainda guarda dois números que a busca usa depois:

- **`nitidez`** — variância do Laplaciano do recorte. Baixo = borrado;
- **`area_rel`** — fração do quadro ocupada pelo rosto. Abaixo de ~0.3% é
  gente de multidão, cujo vetor tende ao "rosto médio" e casa fraco com todo
  mundo.

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

Não precisa reindexar. O `indexar.py` descreve o *acervo*, não as pessoas: os
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
| `--db ARQ` | `solenidade.db` — índice do acervo, feito pelo `indexar.py` |
| `--memoria ARQ` | `pessoas.db` — cadastro de rostos entre acervos |

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

Alvo que está no acervo aparece com melhor acerto entre 0.01 e 0.25 e um
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
recorte de 185×192 tirado do próprio acervo foi a 0.039 com gap limpo.

## A pipeline completa

Só é necessária uma vez por acervo — depois disso, o dia a dia é o `alvo`.

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

**Busca e agrupamento respondem perguntas diferentes.** A busca parte de *quem
você procura* e precisa de uma foto de referência. O agrupamento não precisa de
referência nenhuma: ele junta os rostos por semelhança e devolve grupos
anônimos, para você descobrir *quem está no acervo* — inclusive quem você não
esperava.

O passo 4 é manual **de propósito**. Construir uma interface de revisão foi
avaliado em uma semana de trabalho para ganho nenhum: o Finder em modo galeria
já mostra as folhas de contato lado a lado, e renomear arquivo é rápido.

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
python indexar.py ~/Fotos/OutroAcervo --db outro.db     # indexa o acervo novo
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
qualquer um deles — **não a média**. A diferença importa: a média de um vetor
frontal com um de perfil produz um borrão que não casa bem com nenhum dos dois,
enquanto a menor distância deixa cada ângulo guardado cobrir a sua própria
pose. Isso resolve o problema de perfil, porque o ArcFace é treinado em rostos
frontais e penaliza vista lateral com força.

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

Acerto aqui é trabalho de limiar, não de código.

| Sintoma | O que fazer |
|---|---|
| Poucos rostos detectados em fotos de formação | `indexar.py --det-size 1600` (reindexar, ~2x mais lento) |
| Mesma pessoa em vários clusters | `agrupar.py --limiar 1.05` |
| Pessoas diferentes no mesmo cluster | `agrupar.py --limiar 0.75` |
| Busca perdendo fotos | `alvo NOME --limiar 0.55`, ou `--realimentar 12` |
| Busca perdendo fotos de perfil | `--realimentar` resolve melhor que subir o limiar |
| Busca trazendo gente errada | `alvo NOME --limiar 0.35` |
| Referência não detecta rosto nenhum | recorte menos justo, ou uma foto do próprio acervo |
| Muito rosto borrado/minúsculo | `agrupar.py --nitidez-min 60 --area-min 0.002` |
| Entregar só fotos onde a pessoa é o assunto | `distribuir.py --area-min 0.01` |

## Regras

- A pasta de origem é **somente leitura**. Nada é movido, renomeado ou
  apagado. Nunca.
- Saída usa **hardlink** — 50 pessoas × 100 fotos custa ~0 bytes. O que o `du`
  mostra é o mesmo dado contado duas vezes, não espaço novo. Só use `--copiar`
  no momento da entrega (pendrive, zip). Destino em outro sistema de arquivos
  cai para cópia sozinho.
- O prefixo numérico no nome do arquivo é a distância: os primeiros arquivos da
  pasta são os mais confiáveis. **Confira de baixo para cima**, que é onde mora
  o erro.
- **A decisão do `alvo` é 100% numérica.** Ele garante que o conjunto está
  separado do resto, não que seja a pessoa certa. Espere 70–85% de acerto,
  não 99% — e diga isso a quem recebe, em vez de sugerir o contrário.

## LGPD

O índice do acervo e o `pessoas.db` contêm **dado biométrico** (Lei
13.709/2018, art. 5º, II). O `pessoas.db` ainda associa o vetor a um **nome**,
o que o torna mais sensível que o índice.

Não versione, não suba pra nuvem, não compartilhe — nem como contexto para
ferramenta remota. Guarde offline ou apague depois da entrega — mas note que
ele é o que torna o próximo acervo barato (as pessoas já ficam reconhecidas).
Se for guardar, vale alinhar com quem responde pela operação.

> O `.gitignore` deste diretório cobre `Fotos/`, `Alvos/` e `.venv/`, mas
> **não** cobre `*.db`. Confira antes de qualquer `git add .`.
