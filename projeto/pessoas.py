#!/usr/bin/env python3
"""
pessoas.py - Consulta e manutencao do cadastro de rostos (pessoas.db).

    python pessoas.py                          # lista quem esta cadastrado
    python pessoas.py --renomear "Cel-Tiberio" "Cel Tiberio"
    python pessoas.py --remover "Fulano"       # apaga a pessoa e seus vetores
    python pessoas.py --esquecer-matches "X"   # so os vetores de --realimentar

Quem cria o cadastro e o buscar.py; aqui so se olha e se arruma.
"""
import argparse

import memoria


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memoria", default=memoria.PADRAO)
    ap.add_argument("--remover", metavar="NOME")
    ap.add_argument("--renomear", nargs=2, metavar=("ANTIGO", "NOVO"))
    ap.add_argument("--esquecer-matches", metavar="NOME",
                    help="descarta os vetores vindos de --realimentar, "
                         "preservando as referencias originais")
    args = ap.parse_args()

    con = memoria.abrir(args.memoria)

    if args.remover:
        pid = con.execute("SELECT id FROM pessoas WHERE nome=?", (args.remover,)).fetchone()
        if not pid:
            raise SystemExit(f"'{args.remover}' nao esta no cadastro.")
        con.execute("DELETE FROM referencias WHERE pessoa_id=?", (pid[0],))
        con.execute("DELETE FROM pessoas WHERE id=?", (pid[0],))
        con.commit()
        print(f"'{args.remover}' removido.")

    if args.renomear:
        antigo, novo = args.renomear
        cur = con.execute("UPDATE pessoas SET nome=? WHERE nome=?", (novo, antigo))
        con.commit()
        if not cur.rowcount:
            raise SystemExit(f"'{antigo}' nao esta no cadastro.")
        print(f"'{antigo}' -> '{novo}'")

    if args.esquecer_matches:
        pid = con.execute("SELECT id FROM pessoas WHERE nome=?",
                          (args.esquecer_matches,)).fetchone()
        if not pid:
            raise SystemExit(f"'{args.esquecer_matches}' nao esta no cadastro.")
        cur = con.execute(
            "DELETE FROM referencias WHERE pessoa_id=? AND fonte='match'", (pid[0],)
        )
        con.commit()
        print(f"{cur.rowcount} vetor(es) de match descartado(s).")

    linhas = memoria.resumo(con)
    if not linhas:
        print(f"Cadastro {args.memoria} vazio.")
        return
    print(f"{len(linhas)} pessoa(s) em {args.memoria}\n")
    print(f"  {'NOME':<28} {'REFS':>5} {'MATCHES':>8}  CADASTRADA EM")
    for nome, refs, matches, criado in linhas:
        print(f"  {nome:<28} {refs or 0:>5} {matches or 0:>8}  {criado}")


if __name__ == "__main__":
    main()
