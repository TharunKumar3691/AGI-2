"""CPU program search. A program is kept only if it reproduces EVERY train pair exactly.

Each candidate is a grid function f (possibly parametrised); a consistent post colour-map is
learned on top of it, so f only has to get the geometry right.
"""
import itertools
import time

import numpy as np
from scipy.ndimage import label as cc_label

from arcio import d8_fwd

ST4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
ST8 = np.ones((3, 3), dtype=int)


def A(g):
    return np.asarray(g, dtype=np.int64)


def bg_of(a):
    v, n = np.unique(a, return_counts=True)
    return int(v[np.argmax(n)])


# ----------------------------------------------------------------------------- primitives
def objects(a, bg, diag, multicolor):
    out = []
    st = ST8 if diag else ST4
    if multicolor:
        lab, n = cc_label(a != bg, structure=st)
        for i in range(1, n + 1):
            out.append(lab == i)
    else:
        for c in np.unique(a):
            if c == bg:
                continue
            lab, n = cc_label(a == c, structure=st)
            for i in range(1, n + 1):
                out.append(lab == i)
    return out


def bbox(mask):
    ys, xs = np.where(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def crop_mask(a, mask, keep_only=False, bg=0):
    y0, y1, x0, x1 = bbox(mask)
    sub = a[y0:y1, x0:x1].copy()
    if keep_only:
        sub[~mask[y0:y1, x0:x1]] = bg
    return sub


SELECTORS = {
    "largest": lambda objs, a: max(range(len(objs)), key=lambda i: (objs[i].sum(), -bbox(objs[i])[0])),
    "smallest": lambda objs, a: min(range(len(objs)), key=lambda i: (objs[i].sum(), bbox(objs[i])[0])),
    "largest_bbox": lambda objs, a: max(range(len(objs)), key=lambda i: _bbox_area(objs[i])),
    "most_colours": lambda objs, a: max(range(len(objs)), key=lambda i: len(np.unique(a[objs[i]]))),
    "unique_colour": lambda objs, a: _unique_by(objs, a, lambda m: tuple(sorted(np.unique(a[m])))),
    "unique_shape": lambda objs, a: _unique_by(objs, a, lambda m: _shape_sig(m)),
    "unique_size": lambda objs, a: _unique_by(objs, a, lambda m: int(m.sum())),
    "top_left": lambda objs, a: min(range(len(objs)), key=lambda i: bbox(objs[i])[:3:2]),
    "bottom_right": lambda objs, a: max(range(len(objs)), key=lambda i: (bbox(objs[i])[1], bbox(objs[i])[3])),
    "densest": lambda objs, a: max(range(len(objs)), key=lambda i: objs[i].sum() / _bbox_area(objs[i])),
    "sparsest": lambda objs, a: min(range(len(objs)), key=lambda i: objs[i].sum() / _bbox_area(objs[i])),
}


def _bbox_area(m):
    y0, y1, x0, x1 = bbox(m)
    return (y1 - y0) * (x1 - x0)


def _shape_sig(m):
    y0, y1, x0, x1 = bbox(m)
    return m[y0:y1, x0:x1].tobytes() + bytes([y1 - y0, x1 - x0])


def _unique_by(objs, a, key):
    keys = [key(m) for m in objs]
    uniq = [i for i, k in enumerate(keys) if keys.count(k) == 1]
    if len(uniq) != 1:
        raise ValueError("no unique object")
    return uniq[0]


def split_by_lines(a):
    """Split on full rows/cols of one colour; returns list of equal-shape parts or None."""
    H, W = a.shape
    for c in np.unique(a):
        rows = [i for i in range(H) if (a[i] == c).all()]
        cols = [j for j in range(W) if (a[:, j] == c).all()]
        if not rows and not cols:
            continue
        rs = _runs(set(rows), H)
        cs = _runs(set(cols), W)
        parts = [a[r0:r1, c0:c1] for r0, r1 in rs for c0, c1 in cs]
        if len(parts) >= 2 and len({p.shape for p in parts}) == 1:
            return parts
    return None


def _runs(banned, n):
    out, s = [], None
    for i in range(n):
        if i in banned:
            if s is not None:
                out.append((s, i))
                s = None
        elif s is None:
            s = i
    if s is not None:
        out.append((s, n))
    return out


def equal_parts(a, k, axis):
    n = a.shape[axis]
    if n % k:
        return None
    return np.split(a, k, axis=axis)


def gravity(a, d, bg):
    b = np.rot90(a, d)
    out = np.full_like(b, bg)
    for j in range(b.shape[1]):
        col = b[:, j][b[:, j] != bg]
        if len(col):
            out[b.shape[0] - len(col):, j] = col
    return np.rot90(out, -d)


def symmetrize(a, mask_colour, kinds):
    out = a.copy()
    holes = out == mask_colour
    for _ in range(2):
        for k in kinds:
            t = {"h": np.fliplr, "v": np.flipud, "t": np.transpose, "r2": lambda z: np.rot90(z, 2)}[k](out)
            if t.shape != out.shape:
                continue
            fill = holes & (t != mask_colour)
            out[fill] = t[fill]
            holes = out == mask_colour
    if holes.any():
        raise ValueError("unfilled")
    return out


# ----------------------------------------------------------------------------- candidate programs
def programs(train):
    """Yield (name, f) candidates plausible for this task (cheap pre-filtering by shapes)."""
    ins = [A(p["input"]) for p in train]
    outs = [A(p["output"]) for p in train]
    same = all(i.shape == o.shape for i, o in zip(ins, outs))

    for t in range(8):
        yield f"d8_{t}", (lambda a, t=t: d8_fwd(a, t))

    # tilings of D8 copies: output = (ry x rx) blocks
    ratios = {(o.shape[0] / i.shape[0], o.shape[1] / i.shape[1]) for i, o in zip(ins, outs)}
    if len(ratios) == 1:
        ry, rx = next(iter(ratios))
        if ry == int(ry) and rx == int(rx) and ry * rx > 1 and ry <= 5 and rx <= 5:
            ry, rx = int(ry), int(rx)
            yield f"upscale_{ry}x{rx}", (lambda a, ry=ry, rx=rx: np.kron(a, np.ones((ry, rx), dtype=np.int64)))
            yield "kron_self", (lambda a, ry=ry, rx=rx: np.kron((a != bg_of(a)).astype(np.int64), a)
                                if a.shape == (ry, rx) else None)
            # choose a D8 transform per block independently (learned from the first pair)
            i0, o0 = ins[0], outs[0]
            h, w = i0.shape
            per_block = []
            for by in range(ry):
                for bx in range(rx):
                    blk = o0[by * h:(by + 1) * h, bx * w:(bx + 1) * w]
                    opts = [t for t in range(8) if d8_fwd(i0, t).shape == blk.shape and np.array_equal(d8_fwd(i0, t), blk)]
                    per_block.append(opts)
            if all(per_block):
                for choice in itertools.islice(itertools.product(*per_block), 16):
                    def tile(a, choice=choice, ry=ry, rx=rx):
                        rows = [np.concatenate([d8_fwd(a, choice[by * rx + bx]) for bx in range(rx)], 1) for by in range(ry)]
                        return np.concatenate(rows, 0)
                    yield f"tile_{choice}", tile
        if 1 / ry == int(1 / ry) and 1 / rx == int(1 / rx) and (ry < 1 or rx < 1):
            fy, fx = int(round(1 / ry)), int(round(1 / rx))
            yield f"downscale_{fy}x{fx}", (lambda a, fy=fy, fx=fx: a[::fy, ::fx])

    if not same:
        # crops: to content bbox, to selected objects
        yield "crop_content", (lambda a: crop_mask(a, a != bg_of(a)))
        for diag in (False, True):
            for multi in (True, False):
                for sel_name, sel in SELECTORS.items():
                    for keep in (False, True):
                        def f(a, diag=diag, multi=multi, sel=sel, keep=keep):
                            bg = bg_of(a)
                            objs = objects(a, bg, diag, multi)
                            if not objs or len(objs) > 60:
                                return None
                            return crop_mask(a, objs[sel(objs, a)], keep, bg)
                        yield f"obj_{sel_name}_{diag}_{multi}_{keep}", f
        # boolean combination of parts (split by separator lines or into equal halves)
        for splitter_name, splitter in [("lines", split_by_lines)] + [
                (f"eq{k}{ax}", (lambda a, k=k, ax=ax: equal_parts(a, k, ax))) for k in (2, 3) for ax in (0, 1)]:
            for op in ("and", "or", "xor", "nor", "a_not_b", "b_not_a"):
                def f(a, splitter=splitter, op=op):
                    parts = splitter(a)
                    if parts is None or len(parts) < 2:
                        return None
                    bgs = [bg_of(p) for p in parts]
                    m = [p != 0 for p in parts] if 0 in a else [p != b for p, b in zip(parts, bgs)]
                    x, y = m[0], m[1]
                    for z in m[2:]:
                        x = x | z if op == "or" else x & z if op == "and" else x ^ z if op == "xor" else x
                    r = {"and": x & y, "or": x | y, "xor": x ^ y, "nor": ~(x | y), "a_not_b": x & ~y,
                         "b_not_a": y & ~x}[op]
                    return r.astype(np.int64)
                yield f"bool_{splitter_name}_{op}", f
        # choose one part by property
        for splitter_name, splitter in [("lines", split_by_lines)]:
            for sel_name, key in [("most_fg", lambda p: (p != bg_of(p)).sum()), ("least_fg", lambda p: -(p != bg_of(p)).sum()),
                                  ("most_colours", lambda p: len(np.unique(p))), ("least_colours", lambda p: -len(np.unique(p)))]:
                def f(a, splitter=splitter, key=key):
                    parts = splitter(a)
                    if parts is None:
                        return None
                    ks = [key(p) for p in parts]
                    if ks.count(max(ks)) != 1:
                        return None
                    return parts[int(np.argmax(ks))]
                yield f"part_{splitter_name}_{sel_name}", f
            def f(a, splitter=splitter):
                parts = splitter(a)
                if parts is None:
                    return None
                sigs = [p.tobytes() for p in parts]
                odd = [i for i, s in enumerate(sigs) if sigs.count(s) == 1]
                return parts[odd[0]] if len(odd) == 1 else None
            yield f"part_{splitter_name}_odd", f
    else:
        for d in range(4):
            yield f"gravity_{d}", (lambda a, d=d: gravity(a, d, bg_of(a)))
        cols = set(np.unique(np.concatenate([i.ravel() for i in ins]))) - set(np.unique(np.concatenate([o.ravel() for o in outs])))
        for mc in cols:
            for kinds in (("h",), ("v",), ("h", "v"), ("t",), ("h", "v", "t"), ("r2",), ("h", "v", "t", "r2")):
                yield f"sym_{mc}_{kinds}", (lambda a, mc=mc, kinds=kinds: symmetrize(a, mc, kinds))


def learn_colour_map(preds, outs):
    cm = {}
    for p, o in zip(preds, outs):
        if p is None or p.shape != o.shape:
            return None
        for a, b in zip(p.ravel().tolist(), o.ravel().tolist()):
            if cm.setdefault(a, b) != b:
                return None
    return cm


def apply_cm(g, cm):
    lut = np.arange(max(10, g.max() + 1))
    for a, b in cm.items():
        if a < len(lut):
            lut[a] = b
    return lut[g]


def solve(task, time_limit=20.0, max_preds=2):
    """Returns per test input a list of up to `max_preds` predicted grids (as lists)."""
    t0 = time.time()
    train = task["train"]
    ins = [A(p["input"]) for p in train]
    outs = [A(p["output"]) for p in train]
    tests = [A(t["input"]) for t in task["test"]]
    found = [[] for _ in tests]
    if all(np.array_equal(o, outs[0]) for o in outs[1:]):
        return [[outs[0].tolist()] for _ in tests], ["constant"]
    names = []
    for name, f in programs(train):
        if time.time() - t0 > time_limit:
            break
        try:
            preds = [f(i) for i in ins]
            if any(p is None for p in preds):
                continue
            preds = [A(p) for p in preds]
            cm = learn_colour_map(preds, outs)
            if cm is None:
                continue
            if all(np.array_equal(i, o) for i, o in zip(ins, outs)):
                continue
            res = []
            for t in tests:
                p = f(t)
                if p is None:
                    raise ValueError
                p = apply_cm(A(p), cm)
                if p.ndim != 2 or not (1 <= p.shape[0] <= 30 and 1 <= p.shape[1] <= 30) or p.max() > 9:
                    raise ValueError
                res.append(p)
        except Exception:
            continue
        added = False
        for ti, p in enumerate(res):
            if not any(np.array_equal(p, q) for q in found[ti]) and len(found[ti]) < max_preds:
                found[ti].append(p)
                added = True
        if added:
            names.append(name)
        if all(len(f_) >= max_preds for f_ in found):
            break
    return [[g.tolist() for g in f_] for f_ in found], names


def solve_many(tasks, time_limit=20.0):
    out = {}
    for k, t in tasks.items():
        try:
            preds, names = solve(t, time_limit)
            if any(preds):
                out[k] = {"preds": preds, "programs": names}
        except Exception:
            pass
    return out
