#!/usr/bin/env python3
"""
agrupar.py - Separa todos por ocorrencia de rosto, sem referencia nenhuma.

    python agrupar.py --db solenidade.db

Gera ./clusters/cluster_000.png ... : uma folha de contato por grupo, com os
rostos daquele grupo em grade.

FLUXO MANUAL (de proposito - construir UI de revisao custa uma semana):
  1. abra a pasta ./clusters no Finder, em modo galeria
  2. renomeie os arquivos que voce reconhecer, mantendo o prefixo:
        cluster_007.png  ->  cluster_007 - Cap Silva.png
  3. deixe intocado o que nao reconhecer / estiver misturado
  4. rode distribuir.py

Ajuste --limiar se os grupos estiverem fragmentados (aumente) ou
misturando pessoas (diminua).
"""
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

MINIATURA = 128
COLUNAS = 8
MAX_POR_FOLHA = 48


def carregar(db, nitidez_min, area_min):
    con = sqlite3.connect(db)
    linhas = con.execute(
        "SELECT r.id, r.emb, r.det_score, r.nitidez, r.area_rel,"
        "       r.x1, r.y1, r.x2, r.y2, f.caminho "
        "FROM rostos r JOIN fotos f ON f.id = r.foto_id "
        "WHERE r.nitidez >= ? AND r.area_rel >= ?",
        (nitidez_min, area_min),
    ).fetchall()
    if not linhas:
        raise SystemExit("Nada passou nos filtros. Baixe --nitidez-min / --area-min.")
    embs = np.vstack([np.frombuffer(l[1], dtype=np.float32) for l in linhas])
    meta = [
        {
            "id": l[0], "det": l[2], "nitidez": l[3],
            "box": (l[5], l[6], l[7], l[8]), "caminho": l[9],
        }
        for l in linhas
    ]
    return embs, meta


def folha_de_contato(itens, destino):
    from PIL import Image, ImageOps

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from indexar import carregar_imagem_bgr  # noqa: F401  (registra HEIC)

    crops = []
    for it in itens[:MAX_POR_FOLHA]:
        try:
            img = Image.open(it["caminho"])
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        x1, y1, x2, y2 = it["box"]
        mx, my = (x2 - x1) * 0.25, (y2 - y1) * 0.25  # margem: contexto ajuda
        caixa = (
            max(0, int((x1 - mx) * w)), max(0, int((y1 - my) * h)),
            min(w, int((x2 + mx) * w)), min(h, int((y2 + my) * h)),
        )
        crops.append(img.crop(caixa).resize((MINIATURA, MINIATURA), Image.LANCZOS))

    if not crops:
        return False
    linhas = (len(crops) + COLUNAS - 1) // COLUNAS
    folha = Image.new("RGB", (COLUNAS * MINIATURA, linhas * MINIATURA), (20, 20, 20))
    for i, c in enumerate(crops):
        folha.paste(c, ((i % COLUNAS) * MINIATURA, (i // COLUNAS) * MINIATURA))
    folha.save(destino, quality=88)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="solenidade.db")
    ap.add_argument("--clusters", default="clusters")
    ap.add_argument("--limiar", type=float, default=0.90,
                    help="distancia euclidiana entre vetores normalizados. "
                         "0.90 ~ cosseno 0.40")
    ap.add_argument("--min-grupo", type=int, default=3)
    ap.add_argument("--nitidez-min", type=float, default=25.0)
    ap.add_argument("--area-min", type=float, default=0.0004)
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN

    embs, meta = carregar(args.db, args.nitidez_min, args.area_min)
    print(f"{len(embs)} rostos apos filtro de qualidade. Agrupando...")

    # Vetores normalizados: distancia euclidiana e monotonica com a cosseno,
    # entao da pra usar a arvore do HDBSCAN e evitar a matriz NxN.
    modelo = HDBSCAN(
        min_cluster_size=args.min_grupo,
        min_samples=2,
        metric="euclidean",
        cluster_selection_epsilon=args.limiar,
    )
    rotulos = modelo.fit_predict(embs.astype(np.float64))

    grupos = defaultdict(list)
    for r, m in zip(rotulos, meta):
        grupos[int(r)].append(m)

    ruido = len(grupos.pop(-1, []))
    ordenados = sorted(grupos.values(), key=len, reverse=True)
    print(f"{len(ordenados)} grupos | {ruido} rostos sem grupo\n")

    destino = Path(args.clusters).expanduser().resolve()
    destino.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    con.execute("DROP TABLE IF EXISTS grupos")
    con.execute("CREATE TABLE grupos (rosto_id INTEGER PRIMARY KEY, grupo INTEGER)")

    for n, itens in enumerate(ordenados):
        itens.sort(key=lambda m: (-m["det"], -m["nitidez"]))
        con.executemany(
            "INSERT INTO grupos VALUES (?,?)", [(m["id"], n) for m in itens]
        )
        arq = destino / f"cluster_{n:03d}.jpg"
        if folha_de_contato(itens, arq):
            print(f"  cluster_{n:03d}  {len(itens):4d} rostos")
    con.commit()

    print(f"\nFolhas de contato em {destino}")
    print("Renomeie 'cluster_007.jpg' -> 'cluster_007 - Cap Silva.jpg' e rode distribuir.py")


if __name__ == "__main__":
    main()
