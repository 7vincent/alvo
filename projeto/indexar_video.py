#!/usr/bin/env python3
"""
indexar_video.py - Acha rostos nos videos e grava os vetores num SQLite.

    python indexar_video.py ../Videos --db videos.db

Mesma ideia do indexar.py, com uma diferenca que muda tudo: a unidade aqui
nao e o quadro, e a TRILHA -- o trecho continuo em que um rosto fica em cena.

Video de 10 min a 30 fps sao 18.000 quadros; no ritmo do indexar.py (0,8
foto/s) isso seria uma tarde inteira. Tres decisoes derrubam para minutos:

  1. amostra 3 quadros por segundo em vez de 30. Quem aparece por menos de
     meio segundo nao e identificacao util de qualquer jeito;
  2. so o quadro amostrado e convertido em imagem (grab/retrieve): o
     decodificador avanca sem montar array nenhum no meio do caminho;
  3. a principal: DETECCAO em todo quadro amostrado, RECONHECIMENTO so nos
     melhores quadros de cada trilha. Alguem 30s em cena a 3 fps da 90
     deteccoes -> 1 trilha -> 3 vetores, nao 90. Detectar e barato perto de
     reconhecer, entao e este corte que paga.

Retomavel como o indexar.py: video ja indexado e pulado, e a gravacao de cada
video e uma transacao so -- queda no meio nao deixa video pela metade.

Os videos NUNCA sao modificados ou movidos.
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexar import DET_SCORE_MIN, LADO_MIN_PX, nitidez  # noqa: E402

EXTENSOES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg",
             ".mts", ".m2ts", ".webm", ".wmv", ".flv", ".3gp"}

# Quantos quadros por segundo sao analisados. Cada fps a mais custa
# proporcionalmente e devolve pouco: o que muda a leitura e a pessoa estar em
# cena, nao o quadro exato.
FPS_AMOSTRA = 3.0

# Lado maior do quadro entregue ao modelo. Nao confundir com o det_size: a
# DETECCAO roda em DET_SIZE (o quadro e reduzido para caber la dentro), mas o
# RECORTE alinhado que vira vetor sai deste quadro aqui. Vale mante-lo maior
# que o det_size -- deteccao rapida, vetor bom.
LADO_MAX = 1280
DET_SIZE = 640

# Trilhas.
IOU_MIN = 0.30          # sobreposicao minima para dizer "e o mesmo rosto"
LACUNA_MAX = 1.0        # segundos sem casar antes de fechar a trilha
VETORES_POR_TRILHA = 3
GAP_MIN = 0.5           # segundos minimos entre dois vetores da mesma trilha
LIMITE_CAND = 150       # candidatos guardados por trilha antes de podar
PODA_CAND = 50          # para quantos a poda reduz


def abrir_db(caminho):
    con = sqlite3.connect(caminho)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos (
            id        INTEGER PRIMARY KEY,
            caminho   TEXT UNIQUE NOT NULL,
            duracao   REAL,
            fps       REAL,
            largura   INTEGER,
            altura    INTEGER,
            n_quadros INTEGER DEFAULT 0,      -- quadros AMOSTRADOS
            n_trilhas INTEGER DEFAULT 0,
            status    TEXT DEFAULT 'ok'
        );
        CREATE TABLE IF NOT EXISTS trilhas (
            id        INTEGER PRIMARY KEY,
            video_id  INTEGER NOT NULL REFERENCES videos(id),
            t_ini     REAL, t_fim REAL,       -- segundos
            n_det     INTEGER,
            t_melhor  REAL,                   -- instante do melhor quadro
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,   -- relativos (0..1)
            det_score REAL,
            nitidez   REAL,
            area_rel  REAL
        );
        CREATE TABLE IF NOT EXISTS vetores (
            id        INTEGER PRIMARY KEY,
            trilha_id INTEGER NOT NULL REFERENCES trilhas(id),
            t         REAL,
            nitidez   REAL,
            area_rel  REAL,
            emb       BLOB NOT NULL           -- float32[512] normalizado
        );
        CREATE INDEX IF NOT EXISTS ix_trilhas_video ON trilhas(video_id);
        CREATE INDEX IF NOT EXISTS ix_vetores_trilha ON vetores(trilha_id);
        """
    )
    con.commit()
    return con


def apagar(con, caminho):
    """Remove um video e tudo que pende dele. Usado pelo --refazer."""
    r = con.execute("SELECT id FROM videos WHERE caminho=?", (str(caminho),)).fetchone()
    if not r:
        return
    con.execute("DELETE FROM vetores WHERE trilha_id IN "
                "(SELECT id FROM trilhas WHERE video_id=?)", (r[0],))
    con.execute("DELETE FROM trilhas WHERE video_id=?", (r[0],))
    con.execute("DELETE FROM videos WHERE id=?", (r[0],))
    con.commit()


def hms(t):
    """Segundos -> 00:01:23.4"""
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def iou(a, b):
    """Sobreposicao de duas caixas (x1,y1,x2,y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def casar(trilhas, dets):
    """Casa deteccoes com trilhas abertas, do melhor par para o pior.

    Guloso e sem Hungarian de proposito: a 3 quadros por segundo e camera de
    solenidade, o rosto mal se move de um quadro amostrado para o seguinte --
    o par obvio ganha por margem larga e o casamento otimo nao paga o custo.
    """
    pares = []
    for i, tr in enumerate(trilhas):
        for j, d in enumerate(dets):
            v = iou(tr["caixa"], d["caixa"])
            if v >= IOU_MIN:
                pares.append((v, i, j))
    pares.sort(reverse=True)
    vistas, vistos, casados = set(), set(), []
    for _, i, j in pares:
        if i in vistas or j in vistos:
            continue
        vistas.add(i)
        vistos.add(j)
        casados.append((i, j))
    return casados, [j for j in range(len(dets)) if j not in vistos]


def qualidade(d):
    """Quanto um quadro presta como vetor. Tamanho manda, nitidez modula.

    Rosto grande e um pouco borrado ainda reconhece; rosto nitido de 20 pixels
    nao. A raiz na nitidez impede que um pico de contraste (fardamento,
    bandeira ao fundo) ganhe de um rosto duas vezes maior.
    """
    return (max(0.0, d["nitidez"]) ** 0.5) * d["area_rel"]


def escolher(cands, k, gap=GAP_MIN):
    """Indices dos k melhores quadros da trilha, espalhados no tempo.

    Dois quadros do mesmo segundo sao praticamente o mesmo vetor; meio segundo
    depois ja e outro angulo. Espalhar e o que faz 3 vetores cobrirem frontal,
    perfil e cabeca baixa em vez de tres frontais identicos -- a mesma razao
    pela qual o cadastro guarda varios vetores por pessoa em vez da media.
    O primeiro da lista e sempre o de maior qualidade.
    """
    ordem = sorted(range(len(cands)), key=lambda i: -qualidade(cands[i]))
    esc = []
    for i in ordem:
        if len(esc) >= k:
            break
        if all(abs(cands[i]["t"] - cands[e]["t"]) >= gap for e in esc):
            esc.append(i)
    for i in ordem:                      # trilha curta: completa sem o gap
        if len(esc) >= k:
            break
        if i not in esc:
            esc.append(i)
    return esc


def analisar(app, caminho, fps_amostra=FPS_AMOSTRA, lado_max=LADO_MAX, quieto=False):
    """Varre o video e devolve (meta, trilhas). Nao toca no banco.

    Quem grava e o chamador, numa transacao so: assim uma queda no meio de um
    video de uma hora nao deixa meio video no indice.
    """
    import cv2
    from insightface.utils import face_align

    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise RuntimeError("nao foi possivel abrir o arquivo")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if not (0.0 < fps <= 240.0):
        fps = 30.0        # container mentindo; 30 e o palpite menos ruim
    passo = max(1, int(round(fps / fps_amostra)))
    alvo_total = (total // passo) if total > 0 else 0

    abertas, fechadas = [], []
    n = n_am = 0
    t = 0.0
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        indice = n
        n += 1
        if indice % passo:
            continue
        ok, quadro = cap.retrieve()
        if not ok:
            continue
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if ms > 0 else indice / fps
        n_am += 1

        h, w = quadro.shape[:2]
        if max(h, w) > lado_max:
            e = lado_max / max(h, w)
            quadro = cv2.resize(quadro, (int(w * e), int(h * e)),
                                interpolation=cv2.INTER_AREA)
            h, w = quadro.shape[:2]

        caixas, kpss = app.det_model.detect(quadro, max_num=0, metric="default")
        dets = []
        for idx in range(len(caixas)):
            x1, y1, x2, y2 = [float(v) for v in caixas[idx][:4]]
            score = float(caixas[idx][4])
            if score < DET_SCORE_MIN or min(x2 - x1, y2 - y1) < LADO_MIN_PX:
                continue
            if kpss is None:
                continue          # sem landmarks nao ha alinhamento, nem vetor
            recorte = quadro[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            dets.append({
                "t": t,
                "caixa": (x1 / w, y1 / h, x2 / w, y2 / h),
                "det_score": score,
                "nitidez": nitidez(recorte),
                "area_rel": ((x2 - x1) * (y2 - y1)) / (w * h),
                # 112x112 alinhado = 37 KB. Guardar o recorte em vez do quadro
                # inteiro e o que permite adiar o reconhecimento ate o fim da
                # trilha sem estourar a memoria.
                "aimg": face_align.norm_crop(quadro, landmark=kpss[idx],
                                             image_size=112),
            })

        casados, novas = casar(abertas, dets)
        for i, j in casados:
            tr = abertas[i]
            tr["t_fim"] = t
            tr["caixa"] = dets[j]["caixa"]
            tr["n_det"] += 1
            tr["cands"].append(dets[j])
            if len(tr["cands"]) > LIMITE_CAND:
                mantidos = sorted(escolher(tr["cands"], PODA_CAND))
                tr["cands"] = [tr["cands"][m] for m in mantidos]
        for j in novas:
            abertas.append({"t_ini": t, "t_fim": t, "caixa": dets[j]["caixa"],
                            "n_det": 1, "cands": [dets[j]]})

        vivas = []
        for tr in abertas:
            (vivas if t - tr["t_fim"] <= LACUNA_MAX else fechadas).append(tr)
        abertas = vivas

        if not quieto and n_am % 100 == 0:
            vel = n_am / max(1e-9, time.time() - t0)
            falta = (f" faltam ~{(alvo_total - n_am) / vel / 60:.1f} min"
                     if alvo_total > n_am else "")
            print(f"      {n_am}/{alvo_total or '?'} quadros  {hms(t)}  "
                  f"{vel:.1f} q/s{falta}", flush=True)

    cap.release()
    fechadas.extend(abertas)
    meta = {"duracao": t, "fps": fps, "largura": larg, "altura": alt,
            "n_quadros": n_am}
    return meta, fechadas


def gravar(con, app, caminho, meta, trilhas, k=VETORES_POR_TRILHA):
    """Reconhece os quadros escolhidos e grava tudo numa transacao so."""
    rec = app.models["recognition"]
    cur = con.execute(
        "INSERT INTO videos(caminho,duracao,fps,largura,altura,n_quadros,"
        "n_trilhas) VALUES(?,?,?,?,?,?,?)",
        (str(caminho), meta["duracao"], meta["fps"], meta["largura"],
         meta["altura"], meta["n_quadros"], len(trilhas)))
    vid = cur.lastrowid
    n_vet = 0
    for tr in trilhas:
        esc = escolher(tr["cands"], k)
        melhor = tr["cands"][esc[0]]
        x1, y1, x2, y2 = melhor["caixa"]
        cur = con.execute(
            "INSERT INTO trilhas(video_id,t_ini,t_fim,n_det,t_melhor,"
            "x1,y1,x2,y2,det_score,nitidez,area_rel)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (vid, tr["t_ini"], tr["t_fim"], tr["n_det"], melhor["t"],
             x1, y1, x2, y2, melhor["det_score"], melhor["nitidez"],
             melhor["area_rel"]))
        tid = cur.lastrowid
        # um lote so por trilha: o blobFromImages da onnxruntime rende bem
        # mais com 3 recortes juntos do que com 3 chamadas
        escolhidos = [tr["cands"][i] for i in sorted(esc)]
        feats = rec.get_feat([d["aimg"] for d in escolhidos])
        for d, f in zip(escolhidos, feats):
            v = np.asarray(f, dtype=np.float32)
            v = v / np.linalg.norm(v)      # normalizado, como no indexar.py
            con.execute(
                "INSERT INTO vetores(trilha_id,t,nitidez,area_rel,emb)"
                " VALUES(?,?,?,?,?)",
                (tid, d["t"], d["nitidez"], d["area_rel"], v.tobytes()))
            n_vet += 1
    con.commit()
    return len(trilhas), n_vet


def indexar(con, videos, app, fps_amostra=FPS_AMOSTRA, lado_max=LADO_MAX,
            k=VETORES_POR_TRILHA, quieto=False):
    """Indexa uma lista de videos ja filtrada. Devolve (trilhas, vetores)."""
    tot_tr = tot_vet = 0
    for i, p in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {p.name}", flush=True)
        try:
            meta, trilhas = analisar(app, p, fps_amostra, lado_max, quieto)
        except Exception as e:
            con.execute("INSERT OR IGNORE INTO videos(caminho,status)"
                        " VALUES(?,?)", (str(p), f"erro_leitura: {e}"))
            con.commit()
            print(f"      ! {e}")
            continue
        n_tr, n_vet = gravar(con, app, p, meta, trilhas, k)
        tot_tr += n_tr
        tot_vet += n_vet
        print(f"      {hms(meta['duracao'])}  {meta['largura']}x{meta['altura']}"
              f"  {meta['fps']:.1f}fps  |  {meta['n_quadros']} quadros  "
              f"{n_tr} trilhas  {n_vet} vetores", flush=True)
    return tot_tr, tot_vet


def main():
    ap = argparse.ArgumentParser(
        description="Indexa os rostos dos videos de uma pasta em trilhas.")
    ap.add_argument("diretorio", help="pasta com os videos (varre recursivamente)")
    ap.add_argument("--db", default="videos.db")
    ap.add_argument("--fps", type=float, default=FPS_AMOSTRA, metavar="N",
                    help=f"quadros por segundo analisados (padrao: {FPS_AMOSTRA:g}). "
                         "Subir so ajuda se a pessoa aparece por menos de 1s")
    ap.add_argument("--det-size", type=int, default=DET_SIZE, metavar="N",
                    help=f"resolucao do detector (padrao: {DET_SIZE}). 1024 acha "
                         "rosto pequeno de plateia e fica ~2x mais lento")
    ap.add_argument("--lado-max", type=int, default=LADO_MAX, metavar="N",
                    help=f"lado maior do quadro analisado (padrao: {LADO_MAX})")
    ap.add_argument("--vetores", type=int, default=VETORES_POR_TRILHA, metavar="N",
                    help=f"vetores guardados por trilha (padrao: {VETORES_POR_TRILHA})")
    ap.add_argument("--refazer", action="store_true",
                    help="reindexa os videos que ja estao no banco")
    ap.add_argument("--coreml", action="store_true",
                    help="ver indexar.py --coreml. Padrao e CPU")
    args = ap.parse_args()

    import cv2  # noqa: F401  (garante erro cedo se faltar)
    from insightface.app import FaceAnalysis

    raiz = Path(args.diretorio).expanduser().resolve()
    if not raiz.is_dir():
        raise SystemExit(f"pasta nao encontrada: {raiz}")
    arquivos = sorted(p for p in raiz.rglob("*")
                      if p.is_file() and p.suffix.lower() in EXTENSOES)
    print(f"{len(arquivos)} video(s) em {raiz}")

    con = abrir_db(args.db)
    if args.refazer:
        for p in arquivos:
            apagar(con, p)
    ja = {r[0] for r in con.execute("SELECT caminho FROM videos")}
    pendentes = [p for p in arquivos if str(p) not in ja]
    print(f"{len(pendentes)} pendente(s) ({len(ja)} ja indexado(s))\n")
    if not pendentes:
        return

    app = FaceAnalysis(
        name="buffalo_l",
        providers=(["CoreMLExecutionProvider", "CPUExecutionProvider"]
                   if args.coreml else ["CPUExecutionProvider"]))
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    t0 = time.time()
    n_tr, n_vet = indexar(con, pendentes, app, args.fps, args.lado_max, args.vetores)
    print(f"\nPronto. {n_tr} trilhas / {n_vet} vetores em {args.db} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
