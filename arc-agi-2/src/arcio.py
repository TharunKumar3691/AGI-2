"""ARC data handling: token encoding, augmentation, submission building.

Grids are encoded directly to token ids (no string tokenization on the hot path).
The prompt layout follows the chat format the Qwen3-4B grid model was trained on:
    <|im_start|>user\n{grid}<|im_end|><|im_start|>assistant\n{grid}<|im_end|>
"""
import glob
import json
import os
import zlib

import numpy as np

MAX_SIDE = 30


# ----------------------------------------------------------------------------- paths
def find_file(name, roots=("/kaggle/input",)):
    for root in roots:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------- grids
def as_grid(a):
    return np.asarray(a, dtype=np.int64)


def is_valid_grid(a):
    return (isinstance(a, np.ndarray) and a.ndim == 2 and 1 <= a.shape[0] <= MAX_SIDE
            and 1 <= a.shape[1] <= MAX_SIDE and a.min() >= 0 and a.max() <= 9)


def grid_key(a):
    a = np.asarray(a)
    return (a.shape, a.astype(np.int8).tobytes())


# D8 symmetry group: index t -> (transpose first?, number of 90-degree rotations)
D8 = [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0), (True, 1), (True, 2), (True, 3)]


def d8_fwd(a, t):
    tr, k = D8[t]
    a = np.asarray(a)
    if tr:
        a = a.T
    return np.rot90(a, k)


def d8_inv(a, t):
    tr, k = D8[t]
    a = np.rot90(np.asarray(a), -k)
    if tr:
        a = a.T
    return a


def d8_swaps_axes(t):
    tr, k = D8[t]
    return tr != (k % 2 == 1)


class Aug:
    """One augmentation: D8 transform + colour permutation + train-example order."""

    __slots__ = ("t", "perm", "inv_perm", "order")

    def __init__(self, t, perm, order):
        self.t = int(t)
        self.perm = np.asarray(perm, dtype=np.int64)
        self.inv_perm = np.argsort(self.perm)
        self.order = list(order)

    def fwd(self, g):
        return self.perm[d8_fwd(as_grid(g), self.t)]

    def inv(self, g):
        return d8_inv(self.inv_perm[as_grid(g)], self.t)

    def __repr__(self):
        return f"Aug(t={self.t}, perm={''.join(map(str, self.perm))}, order={self.order})"


def make_augs(n_train, n_colour, rng, d8=range(8), identity_first=False):
    """len(d8) * n_colour augmentations; colour permutations touch all 10 colours."""
    augs = []
    for c in range(n_colour):
        for t in d8:
            if identity_first and c == 0:
                perm, order = np.arange(10), list(range(n_train))
            else:
                perm, order = rng.permutation(10), rng.permutation(n_train).tolist()
            augs.append(Aug(t, perm, order))
    return augs


# ----------------------------------------------------------------------------- tokens
class TokenMap:
    """Token ids for digits, newline and the chat-format pieces."""

    def __init__(self, digit, nl, user_prefix, mid, end, pad):
        self.digit = np.asarray(digit, dtype=np.int64)
        self.nl = int(nl)
        self.user_prefix = list(user_prefix)   # "<|im_start|>user\n"
        self.mid = list(mid)                   # "<|im_end|><|im_start|>assistant\n"
        self.end = list(end)                   # "<|im_end|>"
        self.pad = int(pad)
        self.eos = self.end[-1]
        self.decode_vocab = {int(d): i for i, d in enumerate(self.digit)}

    @classmethod
    def from_tokenizer(cls, tok):
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        digit = [enc(str(i)) for i in range(10)]
        nl = enc("\n")
        assert all(len(d) == 1 for d in digit) and len(nl) == 1, (digit, nl)
        end = enc("<|im_end|>")
        pad = tok.pad_token_id if tok.pad_token_id is not None else end[-1]
        tm = cls([d[0] for d in digit], nl[0], enc("<|im_start|>user\n"),
                 enc("<|im_end|><|im_start|>assistant\n"), end, pad)
        assert len(tm.end) == 1, tm.end
        return tm

    def grid(self, g):
        g = as_grid(g)
        h, w = g.shape
        body = np.concatenate([self.digit[g], np.full((h, 1), self.nl)], axis=1).reshape(-1)
        return body[:-1].tolist()      # no trailing newline

    def reply(self, g):
        return self.grid(g) + self.end

    def max_reply_len(self):
        return MAX_SIDE * (MAX_SIDE + 1) - 1 + len(self.end)

    def decode_grid(self, ids):
        """Token ids (without the final eos) -> grid, or None if malformed."""
        rows, row = [], []
        for t in ids:
            t = int(t)
            if t == self.nl:
                rows.append(row)
                row = []
            elif t in self.decode_vocab:
                row.append(self.decode_vocab[t])
            else:
                return None
        rows.append(row)
        if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
            return None
        a = np.asarray(rows, dtype=np.int64)
        return a if is_valid_grid(a) else None


def text_format(pairs, query=None):
    """Reference string format (used only to verify TokenMap against the tokenizer)."""
    gs = lambda g: "\n".join("".join(str(int(v)) for v in row) for row in g)
    s = "".join(f"<|im_start|>user\n{gs(p['input'])}<|im_end|><|im_start|>assistant\n{gs(p['output'])}<|im_end|>"
                for p in pairs)
    if query is not None:
        s += f"<|im_start|>user\n{gs(query)}<|im_end|><|im_start|>assistant\n"
    return s


def encode_pairs(tm, pairs, with_labels=True):
    ids, labels = [], []
    for p in pairs:
        head = tm.user_prefix + tm.grid(p["input"]) + tm.mid
        out = tm.reply(p["output"])
        ids += head + out
        labels += [-100] * len(head) + out
    return (ids, labels) if with_labels else ids


def encode_query(tm, g):
    return tm.user_prefix + tm.grid(g) + tm.mid


def augmented_pairs(task, aug):
    return [{"input": aug.fwd(task["train"][i]["input"]), "output": aug.fwd(task["train"][i]["output"])}
            for i in aug.order]


def fit_pairs(tm, pairs, extra_len, max_len):
    """Drop leading demonstration pairs until the sequence fits (keep >= 1)."""
    pairs = list(pairs)
    enc = [encode_pairs(tm, [p], with_labels=False) for p in pairs]
    total = sum(map(len, enc)) + extra_len
    while total > max_len and len(pairs) > 1:
        total -= len(enc.pop(0))
        pairs.pop(0)
    return pairs, total <= max_len


def train_sample(tm, task, aug, max_len):
    pairs, ok = fit_pairs(tm, augmented_pairs(task, aug), 0, max_len)
    if not ok:
        return None
    ids, labels = encode_pairs(tm, pairs)
    return ids, labels


def decode_prompt(tm, task, test_in, aug, max_len):
    q = encode_query(tm, aug.fwd(test_in))
    pairs, ok = fit_pairs(tm, augmented_pairs(task, aug), len(q) + tm.max_reply_len(), max_len)
    if not ok:
        return None
    return encode_pairs(tm, pairs, with_labels=False) + q


def task_tokens(task):
    """Cheap token-count estimate of all demonstration pairs (for the cost model)."""
    n = 0
    for p in task["train"]:
        for g in (p["input"], p["output"]):
            n += len(g) * (len(g[0]) + 1)
        n += 8
    return n


# ----------------------------------------------------------------------------- submission
def empty_submission(tasks):
    return {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in t["test"]] for k, t in tasks.items()}


def validate_submission(sub, tasks):
    assert set(sub) == set(tasks), "task id mismatch"
    for k, t in tasks.items():
        assert isinstance(sub[k], list) and len(sub[k]) == len(t["test"]), k
        for e in sub[k]:
            assert set(e) == {"attempt_1", "attempt_2"}, k
            for a in e.values():
                g = np.asarray(a)
                assert g.ndim == 2 and g.size > 0 and g.dtype.kind == "i", k
    return True


def score_submission(sub, tasks, solutions):
    got, n = 0.0, 0
    for k in tasks:
        if k not in solutions:
            continue
        sols = solutions[k]
        for i, s in enumerate(sols):
            e = sub[k][i]
            got += any(np.array_equal(np.asarray(e[a]), np.asarray(s)) for a in ("attempt_1", "attempt_2")) / len(sols)
        n += 1
    return got, n


def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
