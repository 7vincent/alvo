#!/usr/bin/env python3
"""
buscar.py - "Te mando um rosto, voce acha as fotos dele."

    python buscar.py referencias/cap_silva.jpg --db solenidade.db
    python buscar.py referencias/            --db solenidade.db --limiar 0.42

Cada arquivo de referencia vira uma pasta em ./saida/<nome-do-arquivo>/ com
hardlinks para as fotos onde aquela pessoa aparece. Hardlink = 0 bytes
extras. As fotos originais nao sao tocadas.

Se voce tiver VARIAS fotos da mesma pessoa, coloque-as numa subpasta com o
nome dela: referencias/cap_silva/*.jpg  -> a media dos embeddings e bem mais
robusta que uma foto so.
"""
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import memoria

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# Distancia cosseno. Menor = mais parecido.
#   0.35 -> conservador (perde fotos, quase nao erra)
#   0.45 -> equilibrio
#   0.55 -> agressivo (pega mais, comeca a misturar gente parecida)
LIMIAR_PADRAO = 0.45


def carregar_indice(db):
    con = sqlite3.connect(db)
    linhas = con.execute(
        "SELECT r.emb, f.caminho, r.nitidez FROM rostos r "
        "JOIN fotos f ON f.id = r.foto_id"
    ).fetchall()
    if not linhas:
        raise SystemExit("Indice vazio. Rode indexar.py primeiro.")
    embs = np.vstack([np.frombuffer(l[0], dtype=np.float32) for l in linhas])
    caminhos = [l[1] for l in linhas]
    nitidez = np.array([l[2] for l in linhas], dtype=np.float32)
    return embs, caminhos, nitidez


# Escalas tentadas na referencia, da maior pra menor. O SCRFD casa rostos
# contra ancoras de escala: numa referencia bem recortada, onde o rosto ocupa
# quase todo o quadro, ao reescalar pra 1024 o rosto fica MAIOR que a maior
# ancora e nenhuma dispara -- o detector devolve zero rostos. Nao e falta de
# resolucao (ampliar a imagem nao resolve, porque o tamanho RELATIVO nao muda);
# o que resolve e baixar o det_size.
DET_SIZES_REF = (1024, 640, 480, 320)


def embutir(app, caminho, verboso=True):
    """Devolve o embedding do maior rosto da imagem de referencia."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from indexar import carregar_imagem_bgr

    img, _, _ = carregar_imagem_bgr(caminho)
    for ds in DET_SIZES_REF:
        app.prepare(ctx_id=0, det_size=(ds, ds))
        rostos = app.get(img)
        if rostos:
            if ds != DET_SIZES_REF[0] and verboso:
                print(f"    ({Path(caminho).name}: detectado com det_size={ds})")
            maior = max(
                rostos, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            return np.asarray(maior.normed_embedding, dtype=np.float32)
    return None


def coletar_consultas(alvo):
    """Devolve {nome_pessoa: [arquivos]}."""
    alvo = Path(alvo).expanduser().resolve()
    if alvo.is_file():
        return {alvo.stem: [alvo]}
    consultas = defaultdict(list)
    for p in sorted(alvo.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSOES:
            nome = p.parent.name if p.parent != alvo else p.stem
            consultas[nome].append(p)
    return consultas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("referencia", nargs="?",
                    help="arquivo ou pasta com fotos de referencia. "
                         "Dispensavel com --conhecidos")
    ap.add_argument("--db", default="solenidade.db")
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--limiar", type=float, default=LIMIAR_PADRAO)
    ap.add_argument("--copiar", action="store_true", help="copiar de verdade em vez de hardlink")
    ap.add_argument("--coreml", action="store_true",
                    help="ver indexar.py --coreml. Padrao e CPU")
    ap.add_argument("--memoria", default=memoria.PADRAO,
                    help="cadastro de rostos ja identificados")
    ap.add_argument("--conhecidos", action="store_true",
                    help="busca TODO mundo que ja esta no cadastro. Nao precisa "
                         "de foto de referencia nem carrega o modelo: e instantaneo")
    ap.add_argument("--quem", nargs="+", metavar="NOME",
                    help="com --conhecidos, restringe a estes nomes")
    ap.add_argument("--nao-memorizar", action="store_true",
                    help="nao grava as referencias no cadastro")
    ap.add_argument("--realimentar-nitidez", type=float, default=60.0, metavar="V",
                    help="nitidez minima para um rosto voltar ao cadastro. "
                         "Rosto borrado vira referencia ruim: o vetor tende ao "
                         "'rosto medio' e passa a casar com outros borrados")
    ap.add_argument("--realimentar", type=int, default=0, metavar="N",
                    help="apos a busca, guarda no cadastro N rostos encontrados, "
                         "escolhidos por diversidade de angulo. Melhora muito a "
                         "busca no proximo evento. Use so apos conferir o resultado")
    args = ap.parse_args()

    if not args.conhecidos and not args.referencia:
        raise SystemExit("Informe uma referencia ou use --conhecidos.")

    embs, caminhos, nitidez = carregar_indice(args.db)
    print(f"Indice: {len(embs)} rostos em {len(set(caminhos))} fotos")

    mem = memoria.abrir(args.memoria)

    # {nome: matriz (n,512)}. Com --conhecidos nem carregamos o detector.
    consultas = {}
    if args.conhecidos:
        consultas = memoria.carregar(mem, args.quem)
        if not consultas:
            raise SystemExit(f"Cadastro {args.memoria} vazio. Rode uma busca "
                             "com foto de referencia primeiro.")
        print(f"Cadastro: {len(consultas)} pessoa(s) — {', '.join(sorted(consultas))}")
    else:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=(
                ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                if args.coreml
                else ["CPUExecutionProvider"]
            ),
        )
        app.prepare(ctx_id=0, det_size=(1024, 1024))

        arquivos_por_nome = coletar_consultas(args.referencia)
        if not arquivos_por_nome:
            raise SystemExit("Nenhuma imagem de referencia encontrada.")
        for nome, arquivos in arquivos_por_nome.items():
            pares = [(a, v) for a in arquivos
                     for v in (embutir(app, a),) if v is not None]
            if not pares:
                print(f"  ! {nome}: nenhum rosto detectado na referencia")
                continue
            if not args.nao_memorizar:
                n = memoria.salvar(mem, nome, pares)
                print(f"  + {nome}: {n} referencia(s) nova(s) no cadastro")
            # junta com o que ja havia no cadastro para essa pessoa
            guardados = memoria.carregar(mem, [nome]).get(nome)
            V = np.vstack([v for _, v in pares])
            consultas[nome] = guardados if guardados is not None else V

    raiz_saida = Path(args.saida).expanduser().resolve()
    for nome, V in consultas.items():
        # menor distancia contra QUALQUER vetor da pessoa. Com varios angulos
        # guardados isso acha perfil, que a media dos vetores perderia.
        # clamp: o vetor da propria foto casa consigo mesmo e o float
        # estoura para -1e-8, virando prefixo '-0.000' no nome do arquivo.
        dist = np.maximum(0.0, 1.0 - (embs @ V.T).max(axis=1))
        idx = np.where(dist <= args.limiar)[0]

        # melhor rosto por foto (uma foto pode ter varios rostos)
        melhor = {}
        for i in idx:
            c = caminhos[i]
            if c not in melhor or dist[i] < melhor[c][0]:
                melhor[c] = (float(dist[i]), int(i))
        if args.realimentar and idx.size:
            bons = idx[nitidez[idx] >= args.realimentar_nitidez]
            descartados = idx.size - bons.size
            if bons.size:
                escolhidos = memoria.diversos(embs[bons], args.realimentar)
                n = memoria.salvar(
                    mem, nome,
                    [(f"{caminhos[bons[e]]}#{bons[e]}", embs[bons[e]])
                     for e in escolhidos],
                    fonte="match",
                )
                print(f"  + {nome}: {n} rosto(s) guardado(s) no cadastro"
                      + (f" ({descartados} borrado(s) descartado(s))"
                         if descartados else ""))
            else:
                print(f"  ! {nome}: nenhum rosto nitido o bastante para "
                      f"realimentar (nitidez < {args.realimentar_nitidez:.0f})")
        melhor = {c: d for c, (d, _) in melhor.items()}

        destino = raiz_saida / nome
        destino.mkdir(parents=True, exist_ok=True)
        for c, d in sorted(melhor.items(), key=lambda kv: kv[1]):
            origem = Path(c)
            # prefixo com a distancia: os primeiros arquivos da pasta sao os
            # mais confiaveis, o que torna a conferencia visual muito rapida
            alvo = destino / f"{d:.3f}_{origem.name}"
            if alvo.exists():
                continue
            try:
                if args.copiar:
                    import shutil

                    shutil.copy2(origem, alvo)
                else:
                    alvo.hardlink_to(origem)
            except OSError:
                import shutil

                shutil.copy2(origem, alvo)

        print(f"  {nome}: {len(melhor)} fotos  ({len(V)} vetor(es) no cadastro)")

    print(f"\nResultado em {raiz_saida}")


if __name__ == "__main__":
    main()
