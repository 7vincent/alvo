#!/usr/bin/env python3
"""
alvo.py - "me da um nome, eu acho o corte e monto a pasta."

    python alvo.py jeova
    python alvo.py DR-Carlos --so-analise
    python alvo.py rubens --limiar 0.55

Encadeia o que se faz na mao antes de chamar buscar.py:

  1. acha a referencia em Alvos/ por pedaco do nome
  2. confirma que o detector achou rosto nela -- crop justo devolve zero
     rostos sem erro nenhum, e essa e a falha silenciosa do projeto
  3. mede a distancia do alvo contra todos os rostos indexados
  4. escolhe o --limiar no degrau do histograma, nao no padrao 0.45
  5. cruza com o cadastro: algum rosto escolhido esta mais perto de OUTRA
     pessoa ja registrada?
  6. chama buscar.py com o limiar escolhido

Quem escreve a pasta e sempre o buscar.py: aqui nao se cria nem se apaga foto.
"""
import argparse
import contextlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent   # projeto/
BASE = RAIZ.parent                       # fotos/
ALVOS = BASE / "Alvos"
DB = RAIZ / "solenidade.db"
MEMORIA = RAIZ / "pessoas.db"

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# Rosto a esta distancia ou menos e acerto obvio: vira vetor extra para
# afiar a leitura do histograma (o mesmo efeito do --realimentar, so que
# em memoria e sem sujar o cadastro).
NUCLEO_MAX = 0.35
# Nao aceita corte colado no nucleo. Sem isto o maior degrau e sempre o vao
# entre o nucleo (distancia ~0) e o resto, e o corte descarta acerto bom.
CORTE_MIN = 0.10
# Acima disto e fundo -- gente diferente. Nao ha degrau util la.
CORTE_MAX = 0.85

# A insightface imprime isto com print() cru no stdout: uma vez por modelo do
# pacote buffalo_l (sao 5, dos quais a pipeline usa 2) e uma a cada prepare().
# Nao ha flag na biblioteca para desligar, entao peneiramos por prefixo --
# assim qualquer coisa inesperada que ela imprima continua aparecendo.
RUIDO = ("Applied providers:", "find model:", "set det-size:")


class _Peneira:
    """Escreve no destino tudo que nao comecar com um prefixo de RUIDO."""

    def __init__(self, destino):
        self.destino = destino
        self.resto = ""

    def write(self, txt):
        self.resto += txt
        while "\n" in self.resto:
            linha, self.resto = self.resto.split("\n", 1)
            if not linha.startswith(RUIDO):
                self.destino.write(linha + "\n")
                self.destino.flush()

    def flush(self):
        self.destino.flush()


@contextlib.contextmanager
def sem_ruido(filtrar=True):
    """Cala o barulho de carga de modelo. O FutureWarning vem do scikit-image
    por dentro do face_align da insightface, e sai no stderr, nao no stdout."""
    if not filtrar:
        yield
        return
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=r".*estimate.*deprecated.*")
        antigo, sys.stdout = sys.stdout, _Peneira(sys.stdout)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = antigo


def rodar_buscar(cmd, cwd, filtrar=True):
    """O buscar.py carrega o mesmo modelo, entao repete o mesmo ruido."""
    if not filtrar:
        subprocess.run(cmd, check=True, cwd=cwd)
        return
    amb = dict(os.environ, PYTHONWARNINGS="ignore::FutureWarning")
    proc = subprocess.Popen(cmd, cwd=cwd, env=amb, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for linha in proc.stdout:
        if not linha.startswith(RUIDO):
            sys.stdout.write(linha)
            sys.stdout.flush()
    if proc.wait():
        raise SystemExit(f"buscar.py falhou (codigo {proc.returncode})")


def achar_referencia(padrao, pasta):
    """Arquivo de Alvos/ cujo nome contem `padrao`. Pasta nao conta."""
    p = Path(padrao).expanduser()
    if p.is_file():
        return p.resolve()
    alvo = padrao.lower()
    achados = [
        f for f in sorted(pasta.iterdir())
        if f.is_file() and f.suffix.lower() in EXTENSOES and alvo in f.name.lower()
    ]
    if not achados:
        disp = ", ".join(f.stem for f in sorted(pasta.iterdir())
                         if f.is_file() and f.suffix.lower() in EXTENSOES)
        raise SystemExit(f"Nenhuma referencia com '{padrao}' em {pasta}\n"
                         f"  disponiveis: {disp}")
    if len(achados) > 1:
        raise SystemExit(f"'{padrao}' casa com mais de uma referencia:\n  "
                         + "\n  ".join(f.name for f in achados))
    return achados[0].resolve()


def carregar_indice(db):
    con = sqlite3.connect(db)
    linhas = con.execute(
        "SELECT r.id, r.emb, f.caminho, r.nitidez, r.area_rel FROM rostos r "
        "JOIN fotos f ON f.id = r.foto_id"
    ).fetchall()
    if not linhas:
        raise SystemExit(f"Indice vazio em {db}. Rode indexar.py primeiro.")
    embs = np.vstack([np.frombuffer(l[1], dtype=np.float32) for l in linhas])
    return (embs, [l[0] for l in linhas], [l[2] for l in linhas],
            np.array([l[3] for l in linhas], dtype=np.float32),
            np.array([l[4] for l in linhas], dtype=np.float32))


def degraus(d, quantos=3):
    """Maiores vaos entre distancias consecutivas na faixa util.

    Devolve [(ultima_aceita, largura_do_vao, n_rostos_dentro)], do maior
    vao para o menor. O corte fica na BORDA DE BAIXO do vao.
    """
    ds = np.sort(d)
    ini = int(np.searchsorted(ds, CORTE_MIN))
    fim = int(np.searchsorted(ds, CORTE_MAX))
    if fim - ini < 2:
        return []
    vaos = np.diff(ds[ini:fim])
    out = []
    for k in np.argsort(vaos)[::-1][:quantos]:
        corte = float(ds[ini + int(k)])
        out.append((corte, float(vaos[int(k)]), int(np.sum(d <= corte))))
    return out


def vizinho_de_outro(embs, sel, mem, nome_alvo, d_alvo):
    """Rostos escolhidos que estao MAIS PERTO de outra pessoa do cadastro.

    Nao depende de limiar nenhum: e so comparar quem esta mais perto.
    """
    import memoria

    cad = memoria.carregar(mem)
    cad.pop(nome_alvo, None)
    if not cad or not len(sel):
        return []
    nomes = list(cad)
    D = np.stack([np.maximum(0.0, 1.0 - (embs[sel] @ cad[n].T).max(axis=1))
                  for n in nomes])          # (pessoas, selecionados)
    melhor = D.argmin(axis=0)
    out = []
    for j, i in enumerate(sel):
        if D[melhor[j], j] < d_alvo[i]:
            out.append((int(i), nomes[melhor[j]], float(D[melhor[j], j])))
    return out


def main():
    ap = argparse.ArgumentParser(
        prog="alvo",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="alvo NOME [opcoes]",
        description=(
            "Acha o corte no histograma de distancias e monta a pasta do alvo.\n"
            "\n"
            "NOME e um pedaco do nome do arquivo de referencia em Alvos/, sem\n"
            "extensao e sem ligar para maiuscula: 'jeova' acha prf-jeova.png.\n"
            "Se casar com mais de um, lista e para."),
        epilog=(
            "o que ele faz, em ordem:\n"
            "  1. acha a referencia em Alvos/ por pedaco do nome\n"
            "  2. confirma que o detector achou rosto nela, descendo o det_size\n"
            "     1024 -> 640 -> 480 -> 320 (crop justo devolve zero rostos SEM\n"
            "     erro nenhum: e a falha silenciosa do projeto)\n"
            "  3. mede a distancia do alvo contra todos os rostos indexados\n"
            "  4. escolhe o --limiar no maior degrau do histograma, em vez do\n"
            "     padrao 0.45 do buscar.py\n"
            "  5. pergunta ao cadastro se algum rosto escolhido esta mais perto\n"
            "     de OUTRA pessoa ja registrada\n"
            "  6. chama o buscar.py com o limiar escolhido\n"
            "\n"
            "quem escreve a pasta e sempre o buscar.py; aqui nao se cria nem se\n"
            "apaga foto, e Fotos/ nunca e tocada.\n"
            "\n"
            "exemplos:\n"
            "  alvo jeova                  analisa, acha o corte, monta a pasta\n"
            "  alvo Carlos --so-analise    so o histograma, nao cria nada\n"
            "  alvo rubens --limiar 0.55   forca o corte\n"
            "  alvo tiberio --copiar       copia de verdade, para pendrive\n"
            "\n"
            "lendo a saida:\n"
            "  d(busca)   distancia contra a referencia + o cadastro. E ESTA que\n"
            "             vira --limiar do buscar.py\n"
            "  d(nucleo)  a mesma com os acertos obvios (<= %.2f) somados como\n"
            "             vetor extra. Separa perfil e cabeca baixa do fundo\n"
            "  x          rosto fora do corte\n"
            "  area%%      tamanho do rosto no quadro. Abaixo de ~0.3%% e gente de\n"
            "             multidao: o vetor tende ao rosto medio e casa fraco\n"
            "             com todo mundo\n"
            "\n"
            "os TRES maiores degraus sao impressos de proposito. Quando os dois\n"
            "primeiros estao colados, a escolha do corte volta a ser sua.\n"
            "\n"
            "a decisao e 100%% numerica: garante que o conjunto esta separado do\n"
            "resto, nao que seja a pessoa certa. Confira a pasta DE BAIXO PARA\n"
            "CIMA, que e onde mora o erro." % NUCLEO_MAX))

    ap.add_argument("-h", "-help", "--help", action="help",
                    help="mostra esta ajuda e sai")
    ap.add_argument("alvo", nargs="?", metavar="NOME",
                    help="pedaco do nome do arquivo de referencia em Alvos/")

    g = ap.add_argument_group("corte")
    g.add_argument("--limiar", type=float, metavar="D",
                   help="forca o corte em vez de achar o degrau. Mesmo sentido "
                        "do --limiar do buscar.py: menor = mais conservador")
    g.add_argument("--so-analise", action="store_true",
                   help="mostra o histograma e para. Nao cria pasta nenhuma")
    g.add_argument("--mostrar", type=int, default=45, metavar="N",
                   help="quantos rostos listar (padrao: 45; 0 desliga a lista)")
    g.add_argument("--verbose", "--verboso", action="store_true",
                   help="mostra o barulho da insightface ao carregar os 5 "
                        "modelos do buffalo_l. Serve para diagnosticar carga "
                        "de modelo e queda de det_size; por padrao fica calado")

    g = ap.add_argument_group("entrega")
    g.add_argument("--copiar", action="store_true",
                   help="copia de verdade em vez de hardlink. So faz sentido "
                        "para pendrive ou zip: hardlink custa ~0 bytes")
    g.add_argument("--realimentar", type=int, default=0, metavar="N",
                   help="guarda N rostos encontrados no cadastro, escolhidos "
                        "por diversidade de angulo. SO depois de conferir a "
                        "pasta com o olho: cadastro sujo degrada em silencio "
                        "toda busca futura, e invalida o limiar recem-usado")
    g.add_argument("--nao-memorizar", action="store_true",
                   help="nao grava a referencia no cadastro")

    g = ap.add_argument_group("caminhos")
    g.add_argument("--alvos", default=str(ALVOS), metavar="DIR",
                   help="pasta das referencias E das entregas (padrao: Alvos/)")
    g.add_argument("--db", default=str(DB), metavar="ARQ",
                   help="indice do evento, feito pelo indexar.py "
                        "(padrao: solenidade.db)")
    g.add_argument("--memoria", default=str(MEMORIA), metavar="ARQ",
                   help="cadastro de rostos entre eventos (padrao: pessoas.db)")

    args = ap.parse_args()
    if args.alvo is None:
        ap.print_help()
        return

    import memoria
    sys.path.insert(0, str(RAIZ))
    import buscar

    pasta_alvos = Path(args.alvos).expanduser().resolve()
    ref = achar_referencia(args.alvo, pasta_alvos)
    nome = ref.stem
    print(f"referencia : {ref.name}  ->  pasta {nome}/")

    with sem_ruido(not args.verbose):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(1024, 1024))
        v = buscar.embutir(app, ref)
    if v is None:
        raise SystemExit(
            "!! nenhum rosto detectado na referencia, nem descendo o det_size.\n"
            "   arrume outra foto: de preferencia do proprio evento, mesmo\n"
            "   que pequena -- dominio importa mais que resolucao.")

    embs, ids, caminhos, nitidez, area = carregar_indice(args.db)
    print(f"indice     : {len(embs)} rostos em {len(set(caminhos))} fotos")

    # V = o que o buscar.py vai usar de fato: a referencia + o que ja estiver
    # no cadastro para essa pessoa. Assim o limiar medido aqui vale la.
    mem = memoria.abrir(args.memoria)
    guardados = memoria.carregar(mem, [nome]).get(nome)
    V = v[None, :] if guardados is None else np.vstack([v[None, :], guardados])
    d = np.maximum(0.0, 1.0 - (embs @ V.T).max(axis=1))
    print(f"cadastro   : {len(V)} vetor(es) para {nome}"
          + ("" if guardados is None else f" ({len(guardados)} ja gravados)"))
    print(f"mais perto : {d.min():.3f}")

    # segunda opiniao: o nucleo entra como vetor extra e separa o que a
    # referencia sozinha deixa ambiguo (perfil, cabeca baixa).
    nucleo = np.where(d <= NUCLEO_MAX)[0]
    if nucleo.size:
        d2 = np.maximum(0.0, 1.0 - (embs @ np.vstack([V, embs[nucleo]]).T).max(axis=1))
    else:
        print(f"  ! nenhum rosto abaixo de {NUCLEO_MAX}: referencia fraca ou "
              "pessoa ausente. Lendo so com a referencia.")
        d2 = d

    cands = degraus(d2)
    if args.limiar is not None:
        # limiar seu = limiar do buscar.py: mede em d, nao no espaco do nucleo
        limiar = args.limiar
        sel = np.where(d <= limiar)[0]
        corte2 = float(d2[sel].max()) if sel.size else 0.0
        origem_corte = "forcado por voce"
    elif not cands:
        raise SystemExit("nenhum degrau na faixa util. Rode --so-analise e olhe.")
    else:
        corte2, largura, _ = cands[0]
        origem_corte = f"degrau de {largura:.3f}"
        sel = np.where(d2 <= corte2)[0]
        limiar = float(d[sel].max()) if sel.size else 0.0
    if not sel.size:
        raise SystemExit("nenhum rosto dentro do corte.")

    print(f"\n{'d(busca)':>9} {'d(nucleo)':>10} {'area%':>6} {'nitidez':>8}  arquivo")
    for i in np.argsort(d2)[:args.mostrar]:
        marca = " " if d2[i] <= corte2 else "x"
        print(f"{marca}{d[i]:8.3f} {d2[i]:10.3f} {area[i]*100:6.2f} "
              f"{nitidez[i]:8.1f}  {Path(caminhos[i]).name}")

    print("\nmaiores degraus (o corte fica na borda de baixo):")
    for c, larg, n in cands:
        aqui = " <== escolhido" if abs(c - corte2) < 1e-9 else ""
        print(f"  {c:.3f} -> {c + larg:.3f}   vao {larg:.3f}   {n} rostos{aqui}")

    conflitos = vizinho_de_outro(embs, sel, mem, nome, d)
    if conflitos:
        print(f"\n!! {len(conflitos)} rosto(s) mais perto de outra pessoa do cadastro:")
        for i, outro, dist in conflitos:
            print(f"   {Path(caminhos[i]).name}: {nome} {d[i]:.3f} "
                  f"vs {outro} {dist:.3f}")
    else:
        print("\ncruzamento com o cadastro: nenhum rosto disputado")

    fotos = {caminhos[i] for i in sel}
    extra = {caminhos[i] for i in np.where(d <= limiar)[0]} - fotos
    print(f"\ncorte      : {corte2:.3f} ({origem_corte})")
    print(f"--limiar   : {limiar:.3f}   -> {len(fotos)} foto(s)")
    if extra:
        print(f"  ! o buscar.py trabalha com {len(V)} vetor(es), nao com o nucleo,"
              f"\n    entao {len(extra)} foto(s) a mais entram no fim da pasta:")
        for c in sorted(extra):
            print(f"      {Path(c).name}")

    if args.so_analise:
        print("\n--so-analise: nada foi criado.")
        return

    cmd = [sys.executable, str(RAIZ / "buscar.py"), str(ref),
           "--db", args.db, "--saida", str(pasta_alvos),
           "--limiar", f"{limiar + 1e-4:.6f}", "--memoria", args.memoria]
    if args.copiar:
        cmd.append("--copiar")
    if args.nao_memorizar:
        cmd.append("--nao-memorizar")
    if args.realimentar:
        cmd += ["--realimentar", str(args.realimentar)]
    print("\n$ " + " ".join(cmd[1:]) + "\n")
    sys.stdout.flush()   # o buscar.py escreve no mesmo fd; sem isto ele sai antes
    rodar_buscar(cmd, RAIZ, filtrar=not args.verbose)

    destino = pasta_alvos / nome
    n = len(list(destino.iterdir())) if destino.is_dir() else 0
    print(f"\n{destino}  ({n} arquivos)")
    print("confira DE BAIXO PARA CIMA: o prefixo e a distancia, "
          "o erro mora no fim.")


if __name__ == "__main__":
    main()
