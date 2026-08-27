#!/usr/bin/env python3
"""
distribuir.py - Le os nomes que voce deu as folhas de contato e monta as
pastas por pessoa.

    python distribuir.py --db solenidade.db

Le ./clusters/ procurando arquivos no formato:
    cluster_007 - Cap Silva.jpg     -> vira a pasta saida/Cap Silva/
    cluster_012.jpg                 -> ignorado (nao foi identificado)

Usa hardlink: uma pessoa em 100 fotos nao custa 100x o espaco. Passe
--copiar quando for gerar o pendrive/zip de entrega.
"""
import argparse
import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

PADRAO = re.compile(r"^cluster_(\d+)\s*[-–—]\s*(.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="solenidade.db")
    ap.add_argument("--clusters", default="clusters")
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--copiar", action="store_true")
    ap.add_argument("--area-min", type=float, default=0.0,
                    help="ex: 0.01 entrega so fotos onde a pessoa aparece "
                         "grande, excluindo fotos de formacao com 40 pessoas")
    args = ap.parse_args()

    nomes = {}
    for p in Path(args.clusters).expanduser().resolve().iterdir():
        m = PADRAO.match(p.stem)
        if m:
            nomes[int(m.group(1))] = m.group(2).strip()

    if not nomes:
        raise SystemExit("Nenhuma folha de contato foi renomeada ainda.")
    print(f"{len(nomes)} pessoas identificadas")

    con = sqlite3.connect(args.db)
    try:
        linhas = con.execute(
            "SELECT g.grupo, f.caminho, r.area_rel "
            "FROM grupos g JOIN rostos r ON r.id = g.rosto_id "
            "JOIN fotos f ON f.id = r.foto_id"
        ).fetchall()
    except sqlite3.OperationalError:
        raise SystemExit("Tabela 'grupos' nao existe. Rode agrupar.py primeiro.")

    por_pessoa = defaultdict(set)
    for grupo, caminho, area in linhas:
        if grupo in nomes and area >= args.area_min:
            por_pessoa[nomes[grupo]].add(caminho)

    raiz = Path(args.saida).expanduser().resolve()
    total = 0
    for pessoa, arquivos in sorted(por_pessoa.items()):
        pasta = raiz / re.sub(r'[/\\:*?"<>|]', "_", pessoa)
        pasta.mkdir(parents=True, exist_ok=True)
        for c in sorted(arquivos):
            origem, alvo = Path(c), pasta / Path(c).name
            if alvo.exists():
                continue
            try:
                shutil.copy2(origem, alvo) if args.copiar else alvo.hardlink_to(origem)
            except OSError:
                shutil.copy2(origem, alvo)
            total += 1
        print(f"  {pessoa}: {len(arquivos)} fotos")

    print(f"\n{total} arquivos em {raiz}")
    print("Confira algumas pastas por amostragem ANTES de entregar.")


if __name__ == "__main__":
    main()
