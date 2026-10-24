"""Old vs new town on the same simulated players. Usage: python research/compare_bots.py <old backend dir> <new backend dir>"""
import contextlib, io, sys, importlib
import numpy as np

def run(backend, leak, sessions=60, games=10, same_words=False):
    for k in [k for k in sys.modules if k == "tilt" or k.startswith("tilt.")]:
        del sys.modules[k]
    sys.path.insert(0, backend)
    from tilt import synthetic
    import tilt.synthetic as syn
    rows = []
    for i in range(sessions):
        if same_words:
            syn.LYING = syn.TRUTHFUL
        with contextlib.redirect_stdout(io.StringIO()):
            s = synthetic.play(n_games=games, leakiness=leak, seed=100 + i)
        md = [v for v in s.votes if v.you_mafia]; vd = [v for v in s.votes if not v.you_mafia]
        fin = [g for g in s.games if g.phase == "over"]
        rows.append((sum(v.out_reader for v in md), sum(v.out_blind for v in md), len(md),
                     sum(v.out_reader for v in vd), sum(v.out_blind for v in vd), len(vd),
                     sum((g.winner == "mafia") == g.you_mafia for g in fin), len(fin)))
    sys.path.remove(backend)
    a = np.array(rows).sum(0)
    return {"caught_reader": a[0] / a[2], "caught_blind": a[1] / a[2], "framed_reader": a[3] / a[5],
            "framed_blind": a[4] / a[5], "human_win": a[6] / a[7]}

old, new = sys.argv[1], sys.argv[2]
for label, kw in [("placebo (same words, leak 0)", dict(leak=0.0, same_words=True)), ("leak 1", dict(leak=1.0))]:
    for name, path in [("old bots", old), ("new bots", new)]:
        r = run(path, **kw)
        print(f"{label:30} {name}: caught {r['caught_reader']:.2f} (blind {r['caught_blind']:.2f})  "
              f"framed {r['framed_reader']:.2f} (blind {r['framed_blind']:.2f})  human wins {r['human_win']:.2f}")
