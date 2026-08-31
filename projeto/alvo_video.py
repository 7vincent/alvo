#!/usr/bin/env python3
"""
alvo_video.py - "me da um nome, eu digo em que minuto ele aparece."

    python alvo_video.py cel-sertao
    python alvo_video.py tiberio --so-analise
    python alvo_video.py rubens --limiar 0.60

Irmao do alvo.py: mesma referencia em Alvos/, mesmo cadastro pessoas.db,
mesma escolha de corte pelo degrau do histograma. So que procura nos videos
de Videos/ em vez de nas fotos, e a resposta e um INTERVALO DE TEMPO:

    cerimonia.mp4
       00:01:23.0 -> 00:01:47.3    24.3s   d=0.412

Video ainda nao indexado e indexado na hora, uma vez so -- o videos.db fica
de cache e a segunda busca e instantanea. Nada em Videos/ e modificado.

Escreve em Alvos/<nome>/video/, fora do caminho das fotos que o alvo.py poe
em Alvos/<nome>/, para as duas entregas nao se misturarem:

    momentos/          um quadro por aparicao, com a caixa do rosto desenhada
    <video>.srt        legenda para arrastar no player e conferir na hora
    linha_do_tempo.txt o que foi impresso aqui

Nao existe --realimentar aqui, de proposito: quadro de video e mais borrado e
mais de perfil que foto posada, e vetor ruim no cadastro degrada em silencio
TODA busca futura, inclusive a das fotos. Se quiser guardar alguem a partir
de um video, tire um still bom e passe pelo alvo.py.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alvo  # noqa: E402  (so define constantes e funcoes ao ser importado)

RAIZ = Path(__file__).resolve().parent   # projeto/
BASE = RAIZ.parent                       # fotos/
ALVOS = BASE / "Alvos"
VIDEOS = BASE / "Videos"
DB = RAIZ / "videos.db"
MEMORIA = RAIZ / "pessoas.db"

# Duas trilhas do mesmo video separadas por menos que isto viram uma aparicao
# so. A pessoa vira de lado, o detector perde tres quadros, a trilha quebra em
# duas -- para quem vai assistir, foi uma aparicao continua.
FOLGA = 2.0

# Vao minimo para chamar um degrau de degrau. Abaixo disto nao ha grupo
# separado do resto -- o que quase sempre quer dizer que o alvo nao aparece.
VAO_MIN = 0.08


def hms(t):
    """Segundos -> 00:01:23.4"""
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def hms_arq(t):
    """Segundos -> 00-01-23 (serve como nome de arquivo)."""
    return f"{int(t // 3600):02d}-{int(t % 3600 // 60):02d}-{int(t % 60):02d}"


def hms_srt(t):
    return (f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:"
            f"{t % 60:06.3f}").replace(".", ",")


def degraus(d, quantos=3):
    """Maiores vaos entre distancias consecutivas. Como o alvo.degraus, mas o
    vao pode ATRAVESSAR o teto da faixa util, e nao so caber dentro dele.

    No indice de fotos ha milhares de rostos: o fundo (gente diferente) forma
    um continuo denso que comeca bem antes de 0.85, entao o degrau do alvo cabe
    inteiro na faixa e o alvo.degraus o enxerga. Num video ha dezenas de
    trilhas, nao milhares -- o alvo fica sozinho la embaixo e o fundo so comeca
    perto de 0.9. O vao verdadeiro vai de 0.46 a 0.87, atravessa o teto, e um
    degraus que so olha para dentro da faixa nao ve degrau nenhum.

    Aqui a borda de BAIXO do vao (que e onde o corte cai) precisa estar na
    faixa util; a de cima pode passar.
    """
    ds = np.sort(d)
    ini = int(np.searchsorted(ds, alvo.CORTE_MIN))
    if len(ds) - ini < 2:
        return []
    vaos = np.diff(ds[ini:])
    out = []
    for k in np.argsort(vaos)[::-1]:
        corte = float(ds[ini + int(k)])
        if corte > alvo.CORTE_MAX:
            continue
        out.append((corte, float(vaos[int(k)]), int(np.sum(d <= corte))))
        if len(out) >= quantos:
            break
    return out


def carregar_trilhas(db):
    """Devolve (embs, idx_trilha, trilhas).

    embs e uma linha por VETOR (varios por trilha); idx_trilha[i] diz a qual
    trilha o vetor i pertence. A busca mede vetor a vetor e depois reduz pelo
    minimo -- mesma logica do cadastro, onde uma pessoa tem varios angulos e
    vale o mais proximo, nao a media.
    """
    if not Path(db).exists():
        raise SystemExit(f"{db} nao existe. Nenhum video foi indexado ainda.")
    con = sqlite3.connect(db)
    linhas = con.execute(
        "SELECT v.trilha_id, v.emb, t.t_ini, t.t_fim, t.n_det, t.t_melhor,"
        "       t.x1, t.y1, t.x2, t.y2, t.nitidez, t.area_rel,"
        "       d.caminho, d.duracao "
        "FROM vetores v "
        "JOIN trilhas t ON t.id = v.trilha_id "
        "JOIN videos  d ON d.id = t.video_id "
        "ORDER BY v.trilha_id"
    ).fetchall()
    if not linhas:
        raise SystemExit(f"Indice vazio em {db}. Rode indexar_video.py primeiro.")
    embs = np.vstack([np.frombuffer(l[1], dtype=np.float32) for l in linhas])
    ordem, trilhas = {}, []
    idx = np.empty(len(linhas), dtype=np.int64)
    for i, l in enumerate(linhas):
        if l[0] not in ordem:
            ordem[l[0]] = len(trilhas)
            trilhas.append({
                "t_ini": l[2], "t_fim": l[3], "n_det": l[4], "t_melhor": l[5],
                "caixa": (l[6], l[7], l[8], l[9]),
                "nitidez": l[10], "area_rel": l[11],
                "video": l[12], "duracao": l[13],
            })
        idx[i] = ordem[l[0]]
    return embs, idx, trilhas


def reduzir(d_vet, idx, n):
    """Distancia por trilha = a do vetor mais proximo dela. Devolve (d, qual)."""
    d = np.full(n, np.inf)
    qual = np.zeros(n, dtype=np.int64)
    for i in range(len(d_vet)):
        g = idx[i]
        if d_vet[i] < d[g]:
            d[g] = d_vet[i]
            qual[g] = i
    return d, qual


def fundir(itens, folga=FOLGA):
    """Junta trilhas vizinhas do mesmo video num intervalo so.

    itens: [(t_ini, t_fim, d, t_melhor, n_det, caixa)]
    """
    out = []
    for it in sorted(itens, key=lambda x: x[0]):
        if out and it[0] <= out[-1]["t_fim"] + folga:
            a = out[-1]
            a["t_fim"] = max(a["t_fim"], it[1])
            a["n_trilhas"] += 1
            a["n_det"] += it[4]
            if it[2] < a["d"]:
                a["d"], a["t_melhor"], a["caixa"] = it[2], it[3], it[5]
        else:
            out.append({"t_ini": it[0], "t_fim": it[1], "d": it[2],
                        "t_melhor": it[3], "n_det": it[4], "caixa": it[5],
                        "n_trilhas": 1})
    return out


def extrair(caminho_video, pedidos, pasta):
    """Salva um quadro por aparicao, com a caixa do rosto desenhada.

    Abre o video uma vez so para todos os pedidos. O seek do OpenCV cai no
    keyframe anterior ao instante pedido, entao avancamos quadro a quadro ate
    passar do tempo certo -- meio segundo de decodificacao por still.
    """
    import cv2

    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        return 0
    n = 0
    for t, caixa, nome_arq in pedidos:
        destino = pasta / nome_arq
        if destino.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 0.5) * 1000.0)
        quadro = None
        for _ in range(200):
            ok, q = cap.read()
            if not ok:
                break
            quadro = q
            if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= t:
                break
        if quadro is None:
            continue
        h, w = quadro.shape[:2]
        x1, y1, x2, y2 = caixa
        cv2.rectangle(quadro, (int(x1 * w), int(y1 * h)),
                      (int(x2 * w), int(y2 * h)), (0, 255, 0), max(2, w // 400))
        cv2.imwrite(str(destino), quadro, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1
    cap.release()
    return n


def escrever_srt(destino, nome, aparicoes):
    linhas = []
    for i, a in enumerate(aparicoes, 1):
        linhas += [str(i),
                   f"{hms_srt(a['t_ini'])} --> {hms_srt(a['t_fim'] + 0.5)}",
                   f"{nome} ({a['d']:.3f})", ""]
    destino.write_text("\n".join(linhas), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        prog="alvo_video",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="alvo_video NOME [opcoes]",
        description=(
            "Acha o alvo nos videos de Videos/ e diz em que minuto ele aparece.\n"
            "\n"
            "NOME e um pedaco do nome do arquivo de referencia em Alvos/, sem\n"
            "extensao e sem ligar para maiuscula: 'sertao' acha cel-sertao.png.\n"
            "A mesma foto que voce usa no alvo.py."),
        epilog=(
            "o que ele faz, em ordem:\n"
            "  1. acha a referencia em Alvos/ por pedaco do nome\n"
            "  2. indexa os videos de Videos/ que ainda nao estao no videos.db\n"
            "     (uma vez so: depois disso a busca e instantanea)\n"
            "  3. mede a distancia do alvo contra todas as trilhas indexadas\n"
            "  4. escolhe o --limiar no maior degrau do histograma\n"
            "  5. funde trilhas vizinhas em aparicoes e imprime a linha do tempo\n"
            "  6. salva um quadro de cada aparicao para voce conferir\n"
            "\n"
            "lendo a saida:\n"
            "  d(busca)   distancia contra a referencia + o cadastro\n"
            "  d(nucleo)  a mesma com os acertos obvios somados como vetor extra\n"
            "  dur        quanto tempo a trilha durou\n"
            "  ndet       em quantos quadros amostrados o rosto foi visto.\n"
            "             trilha de 1-2 quadros costuma ser ruido\n"
            "  x          trilha fora do corte\n"
            "\n"
            "ATENCAO ao limiar: quadro de video tem borrao de movimento e muito\n"
            "mais perfil que foto posada, entao TODAS as distancias sobem. O\n"
            "corte que voce achou nas fotos NAO vale aqui -- deixe o degrau\n"
            "decidir, e confira os momentos/ DE BAIXO PARA CIMA.\n"
            "\n"
            "se nao achar ninguem, o alvo pode estar pequeno demais no quadro:\n"
            "  --det-size 1024 --refazer   (mais lento, acha rosto de plateia)"))

    ap.add_argument("-h", "-help", "--help", action="help",
                    help="mostra esta ajuda e sai")
    ap.add_argument("alvo", nargs="?", metavar="NOME",
                    help="pedaco do nome do arquivo de referencia em Alvos/")

    g = ap.add_argument_group("corte")
    g.add_argument("--limiar", type=float, metavar="D",
                   help="forca o corte em vez de achar o degrau")
    g.add_argument("--so-analise", action="store_true",
                   help="mostra o histograma e para. Nao cria pasta nenhuma")
    g.add_argument("--mostrar", type=int, default=40, metavar="N",
                   help="quantas trilhas listar (padrao: 40; 0 desliga)")
    g.add_argument("--folga", type=float, default=FOLGA, metavar="S",
                   help=f"segundos de intervalo que ainda contam como a mesma "
                        f"aparicao (padrao: {FOLGA:g})")
    g.add_argument("--verbose", "--verboso", action="store_true",
                   help="mostra o barulho da insightface ao carregar os modelos")

    g = ap.add_argument_group("entrega")
    g.add_argument("--sem-momentos", action="store_true",
                   help="nao extrai os quadros de conferencia (so a lista)")
    g.add_argument("--nao-memorizar", action="store_true",
                   help="nao grava a referencia no cadastro")

    g = ap.add_argument_group("indexacao (so afeta video ainda nao indexado)")
    g.add_argument("--sem-indexar", action="store_true",
                   help="nao indexa video novo; usa so o que ja esta no banco")
    g.add_argument("--refazer", action="store_true",
                   help="reindexa TODOS os videos, jogando fora o que ja havia")
    g.add_argument("--fps", type=float, default=None, metavar="N",
                   help="quadros por segundo analisados (padrao: 3)")
    g.add_argument("--det-size", type=int, default=None, metavar="N",
                   help="resolucao do detector (padrao: 640)")
    g.add_argument("--lado-max", type=int, default=None, metavar="N",
                   help="lado maior do quadro analisado (padrao: 1280)")

    g = ap.add_argument_group("caminhos")
    g.add_argument("--videos", default=str(VIDEOS), metavar="DIR",
                   help="pasta dos videos (padrao: Videos/)")
    g.add_argument("--alvos", default=str(ALVOS), metavar="DIR",
                   help="pasta das referencias E das entregas (padrao: Alvos/)")
    g.add_argument("--db", default=str(DB), metavar="ARQ",
                   help="indice dos videos (padrao: videos.db)")
    g.add_argument("--memoria", default=str(MEMORIA), metavar="ARQ",
                   help="cadastro de rostos entre eventos (padrao: pessoas.db)")

    args = ap.parse_args()
    if args.alvo is None:
        ap.print_help()
        return

    sys.path.insert(0, str(RAIZ))
    import buscar
    import indexar_video as iv
    import memoria

    fps = args.fps if args.fps else iv.FPS_AMOSTRA
    det_size = args.det_size if args.det_size else iv.DET_SIZE
    lado_max = args.lado_max if args.lado_max else iv.LADO_MAX

    pasta_alvos = Path(args.alvos).expanduser().resolve()
    ref = alvo.achar_referencia(args.alvo, pasta_alvos)
    nome = ref.stem
    print(f"referencia : {ref.name}  ->  Alvos/{nome}/video/")

    # ---- 1. o que falta indexar -------------------------------------------
    pasta_videos = Path(args.videos).expanduser().resolve()
    con = iv.abrir_db(args.db)
    pendentes = []
    if not args.sem_indexar:
        if not pasta_videos.is_dir():
            raise SystemExit(f"pasta de videos nao encontrada: {pasta_videos}\n"
                             f"  crie-a e ponha os videos la, ou use --videos DIR")
        achados = sorted(p for p in pasta_videos.rglob("*")
                         if p.is_file() and p.suffix.lower() in iv.EXTENSOES)
        if not achados:
            raise SystemExit(f"nenhum video em {pasta_videos}\n"
                             f"  extensoes aceitas: "
                             f"{' '.join(sorted(iv.EXTENSOES))}")
        if args.refazer:
            for p in achados:
                iv.apagar(con, p)
        ja = {r[0] for r in con.execute("SELECT caminho FROM videos")}
        pendentes = [p for p in achados if str(p) not in ja]
        print(f"videos     : {len(achados)} em {pasta_videos.name}/  "
              f"({len(pendentes)} a indexar)")

    # ---- 2. modelo: uma carga so, para a referencia e para a indexacao -----
    with alvo.sem_ruido(not args.verbose):
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(1024, 1024))
        v = buscar.embutir(app, ref)     # isto mexe no det_size do app
    if v is None:
        raise SystemExit(
            "!! nenhum rosto detectado na referencia, nem descendo o det_size.\n"
            "   arrume outra foto: de preferencia do proprio evento, mesmo\n"
            "   que pequena -- dominio importa mais que resolucao.")

    if pendentes:
        print(f"\nindexando {len(pendentes)} video(s) a {fps:g} quadros/s "
              f"(det_size {det_size}) -- so desta vez:\n")
        with alvo.sem_ruido(not args.verbose):
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
            iv.indexar(con, pendentes, app, fps, lado_max)
        print()

    # ---- 3. distancias -----------------------------------------------------
    embs, idx, trilhas = carregar_trilhas(args.db)
    n_videos = len({t["video"] for t in trilhas})
    print(f"indice     : {len(trilhas)} trilhas / {len(embs)} vetores "
          f"em {n_videos} video(s)")

    mem = memoria.abrir(args.memoria)
    if not args.nao_memorizar:
        n_novo = memoria.salvar(mem, nome, [(str(ref), v)])
        if n_novo:
            print(f"  + {nome}: {n_novo} referencia(s) nova(s) no cadastro")
    guardados = memoria.carregar(mem, [nome]).get(nome)
    V = v[None, :] if guardados is None else np.vstack([v[None, :], guardados])
    print(f"cadastro   : {len(V)} vetor(es) para {nome}")

    d_vet = np.maximum(0.0, 1.0 - (embs @ V.T).max(axis=1))
    d, qual = reduzir(d_vet, idx, len(trilhas))
    print(f"mais perto : {d.min():.3f}")

    # segunda opiniao: os acertos obvios entram como vetor extra e separam o
    # que a referencia sozinha deixa ambiguo (perfil, cabeca baixa, borrao)
    nucleo = np.where(d_vet <= alvo.NUCLEO_MAX)[0]
    if nucleo.size:
        d2v = np.maximum(0.0, 1.0 - (embs @ np.vstack([V, embs[nucleo]]).T).max(axis=1))
        d2, _ = reduzir(d2v, idx, len(trilhas))
    else:
        print(f"  ! nenhum vetor abaixo de {alvo.NUCLEO_MAX}: referencia fraca, "
              "alvo ausente\n    ou rosto pequeno demais no quadro. Lendo so "
              "com a referencia.")
        d2 = d

    # Sem corte a lista ainda e impressa: morrer antes de mostrar o histograma
    # esconde justamente o numero que diz o que fazer em seguida.
    cands = degraus(d2)
    vazio = np.empty(0, dtype=np.int64)
    sem_corte = ""
    if args.limiar is not None:
        limiar = args.limiar
        sel = np.where(d <= limiar)[0]
        corte2 = float(d2[sel].max()) if sel.size else -1.0
        origem_corte = "forcado por voce"
        if not sel.size:
            sem_corte = f"nenhuma trilha abaixo de {limiar:.3f}."
    elif not cands:
        sem_corte = ("nao ha nem duas trilhas na faixa util.")
        corte2, limiar, sel, origem_corte = -1.0, 0.0, vazio, "-"
    elif cands[0][1] < VAO_MIN:
        sem_corte = (f"o maior vao e de so {cands[0][1]:.3f} (minimo "
                     f"{VAO_MIN:.2f}): nenhum grupo se separa do resto.")
        corte2, limiar, sel, origem_corte = -1.0, 0.0, vazio, "-"
    else:
        corte2, largura, _ = cands[0]
        origem_corte = f"degrau de {largura:.3f}"
        sel = np.where(d2 <= corte2)[0]
        limiar = float(d[sel].max()) if sel.size else 0.0

    # ---- 4. o que foi visto -----------------------------------------------
    if args.mostrar:
        print(f"\n{'d(busca)':>9} {'d(nucleo)':>10} {'area%':>6} {'dur':>7} "
              f"{'ndet':>5}  {'instante':<12} video")
        for i in np.argsort(d2)[:args.mostrar]:
            t = trilhas[i]
            marca = " " if d2[i] <= corte2 else "x"
            print(f"{marca}{d[i]:8.3f} {d2[i]:10.3f} {t['area_rel']*100:6.2f} "
                  f"{t['t_fim'] - t['t_ini']:6.1f}s {t['n_det']:5d}  "
                  f"{hms(t['t_melhor']):<12} {Path(t['video']).name}")

    print("\nmaiores degraus (o corte fica na borda de baixo):")
    for c, larg, n in cands:
        aqui = " <== escolhido" if abs(c - corte2) < 1e-9 else ""
        print(f"  {c:.3f} -> {c + larg:.3f}   vao {larg:.3f}   {n} trilhas{aqui}")

    if sem_corte:
        print(f"\n!! {sem_corte}")
        print("   normalmente significa que o alvo nao aparece nos videos. Se")
        print("   voce acha que aparece, o que costuma resolver:")
        print("     --det-size 1024 --refazer   rosto longe da camera; ~2x mais lento")
        print("     --fps 6 --refazer           aparicao muito curta")
        print("     --limiar 0.60               voce discorda do numero acima")
        print("\nnada foi criado.")
        return

    conflitos = alvo.vizinho_de_outro(embs[qual], sel, mem, nome, d)
    if conflitos:
        print(f"\n!! {len(conflitos)} trilha(s) mais perto de outra pessoa do cadastro:")
        for i, outro, dist in conflitos:
            t = trilhas[i]
            print(f"   {Path(t['video']).name} {hms(t['t_melhor'])}: "
                  f"{nome} {d[i]:.3f} vs {outro} {dist:.3f}")
    else:
        print("\ncruzamento com o cadastro: nenhuma trilha disputada")

    # ---- 5. linha do tempo -------------------------------------------------
    por_video = {}
    for i in sel:
        t = trilhas[i]
        por_video.setdefault(t["video"], []).append(
            (t["t_ini"], t["t_fim"], float(d[i]), t["t_melhor"], t["n_det"],
             t["caixa"]))

    linhas = [f"# {nome}   corte {corte2:.3f} ({origem_corte})   "
              f"--limiar {limiar:.3f}", ""]
    total = 0
    for caminho in sorted(por_video):
        aps = fundir(por_video[caminho], args.folga)
        por_video[caminho] = aps
        total += len(aps)
        dur = next(t["duracao"] for t in trilhas if t["video"] == caminho)
        linhas.append(f"{Path(caminho).name}   ({hms(dur)})")
        for a in aps:
            linhas.append(
                f"   {hms(a['t_ini'])} -> {hms(a['t_fim'])}   "
                f"{a['t_fim'] - a['t_ini']:6.1f}s   d={a['d']:.3f}   "
                f"{a['n_det']} quadros")
        linhas.append("")
    print("\n" + "\n".join(linhas), end="")
    print(f"corte      : {corte2:.3f} ({origem_corte})")
    print(f"--limiar   : {limiar:.3f}   -> {total} aparicao(oes) "
          f"em {len(por_video)} video(s), {len(sel)} trilhas")

    if args.so_analise:
        print("\n--so-analise: nada foi criado.")
        return

    # ---- 6. entrega --------------------------------------------------------
    destino = pasta_alvos / nome / "video"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "linha_do_tempo.txt").write_text("\n".join(linhas), encoding="utf-8")
    for caminho, aps in por_video.items():
        escrever_srt(destino / f"{Path(caminho).stem}.srt", nome, aps)

    if not args.sem_momentos:
        pasta_m = destino / "momentos"
        pasta_m.mkdir(exist_ok=True)
        n = 0
        for caminho, aps in por_video.items():
            # prefixo com a distancia, igual ao buscar.py: a pasta ordenada ja
            # poe o mais confiavel em cima e o erro no fim
            pedidos = [(a["t_melhor"], a["caixa"],
                        f"{a['d']:.3f}_{hms_arq(a['t_melhor'])}_"
                        f"{Path(caminho).stem}.jpg") for a in aps]
            n += extrair(caminho, pedidos, pasta_m)
        print(f"\n{n} quadro(s) em {pasta_m}")

    print(f"\n{destino}")
    print("confira os momentos/ DE BAIXO PARA CIMA: o prefixo e a distancia, "
          "o erro mora no fim.")


if __name__ == "__main__":
    main()
