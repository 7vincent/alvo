#!/usr/bin/env python3
"""
memoria.py - Cadastro persistente dos rostos ja identificados.

Guarda, por pessoa, os vetores de 512 dimensoes usados nas buscas. Serve para
reencontrar a mesma pessoa numa pasta de fotos NOVA sem precisar reunir as
fotos de referencia outra vez:

    python indexar.py ~/Fotos/OutroEvento --db outro.db
    python buscar.py --conhecidos --db outro.db --saida entrega --copiar

Fica num arquivo separado do indice do evento de proposito: solenidade.db
descreve UM evento e e descartavel; pessoas.db atravessa todos e e o que
torna a proxima busca barata.

Uma pessoa tem VARIOS vetores (frontal, perfil, com e sem oculos). A busca usa
a menor distancia entre o rosto candidato e qualquer um deles, entao cada
vetor novo cobre um angulo a mais.
"""
import sqlite3
from datetime import datetime, timezone

import numpy as np

PADRAO = "pessoas.db"


def abrir(caminho=PADRAO):
    con = sqlite3.connect(caminho)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS pessoas (
            id        INTEGER PRIMARY KEY,
            nome      TEXT UNIQUE NOT NULL,
            criado_em TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS referencias (
            id        INTEGER PRIMARY KEY,
            pessoa_id INTEGER NOT NULL REFERENCES pessoas(id),
            origem    TEXT NOT NULL,
            fonte     TEXT NOT NULL DEFAULT 'referencia',  -- ou 'match'
            emb       BLOB NOT NULL                        -- float32[512]
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ix_ref_unica
            ON referencias(pessoa_id, origem);
        """
    )
    con.commit()
    return con


def salvar(con, nome, itens, fonte="referencia"):
    """itens: lista de (origem, vetor float32[512] normalizado). Devolve novos."""
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(
        "INSERT OR IGNORE INTO pessoas(nome,criado_em) VALUES(?,?)", (nome, agora)
    )
    pid = con.execute("SELECT id FROM pessoas WHERE nome=?", (nome,)).fetchone()[0]
    novos = 0
    for origem, v in itens:
        v = np.asarray(v, dtype=np.float32)
        v = v / np.linalg.norm(v)
        cur = con.execute(
            "INSERT OR IGNORE INTO referencias(pessoa_id,origem,fonte,emb)"
            " VALUES(?,?,?,?)",
            (pid, str(origem), fonte, v.tobytes()),
        )
        novos += cur.rowcount
    con.commit()
    return novos


def carregar(con, nomes=None):
    """Devolve {nome: matriz (n,512)} com todos os vetores de cada pessoa."""
    q = (
        "SELECT p.nome, r.emb FROM referencias r "
        "JOIN pessoas p ON p.id = r.pessoa_id"
    )
    par = ()
    if nomes:
        q += " WHERE p.nome IN (%s)" % ",".join("?" * len(nomes))
        par = tuple(nomes)
    out = {}
    for nome, emb in con.execute(q, par):
        out.setdefault(nome, []).append(np.frombuffer(emb, dtype=np.float32))
    return {k: np.vstack(v) for k, v in out.items()}


def resumo(con):
    """[(nome, n_referencia, n_match, criado_em)] ordenado por nome."""
    return con.execute(
        "SELECT p.nome,"
        "       SUM(r.fonte='referencia'), SUM(r.fonte='match'), p.criado_em "
        "FROM pessoas p LEFT JOIN referencias r ON r.pessoa_id = p.id "
        "GROUP BY p.id ORDER BY p.nome"
    ).fetchall()


def diversos(V, k):
    """Escolhe k vetores bem espalhados (max-min): cobre angulos diferentes
    em vez de guardar 90 frontais quase identicos."""
    if len(V) <= k:
        return list(range(len(V)))
    esc = [int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))]
    while len(esc) < k:
        d = np.min(1.0 - V @ V[esc].T, axis=1)
        d[esc] = -1
        esc.append(int(np.argmax(d)))
    return esc
