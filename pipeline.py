"""Loopy hint engine: extract puzzle from a screenshot, solve, check, hint.

Scope: SGT Puzzles "Loopy" on the Cairo pentagonal grid, phone screenshots
at roughly 900-1100 px wide. Edge states read from colors:
yellow = undecided, black = line, light gray = crossed out.
"""
import itertools
import math
import os

import cv2
import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical195

TEMPLATES = os.path.join(os.path.dirname(__file__), "templates.npz")


class ExtractionError(Exception):
    pass


# ---------------------------------------------------------------- extraction

def _masks(im):
    b = im[:, :, 0].astype(int)
    g = im[:, :, 1].astype(int)
    r = im[:, :, 2].astype(int)
    dark = (r < 100) & (g < 100) & (b < 100)
    yellow = (r > 150) & (g > 150) & (b < 120)
    # blank out UI bars: rows that are mostly dark edge-to-edge
    bar_rows = dark.mean(axis=1) > 0.5
    dark[bar_rows, :] = False
    return dark.astype(np.uint8), yellow, bar_rows


def _find_dots(dark):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    er = cv2.erode(dark, k)
    n, _, stats, cent = cv2.connectedComponentsWithStats(er, 8)
    cands = [cent[i] for i in range(1, n)]
    dots = []
    for c in cands:
        xi, yi = int(round(c[0])), int(round(c[1]))
        disk = dark[yi - 4:yi + 5, xi - 4:xi + 5]
        if disk.size and disk.mean() > 0.9:  # solid disk => a dot, not a line bend
            if all(np.hypot(c[0] - d[0], c[1] - d[1]) > 15 for d in dots):
                dots.append((float(c[0]), float(c[1])))
    if len(dots) < 20:
        raise ExtractionError(
            "Couldn't find the puzzle dots. Use an unzoomed screenshot of a "
            "pentagonal-grid Loopy puzzle.")
    return np.array(dots)


def _edge_scale(dots):
    from scipy.spatial import cKDTree
    t = cKDTree(dots)
    dd, _ = t.query(dots, k=2)
    return float(np.median(dd[:, 1]))


def _classify_segment(im, dark, yellow, p0, p1):
    """Return 'line' / 'undecided' / 'excluded' / None for a dot pair."""
    d = np.hypot(*(p1 - p0))
    ts = np.linspace(15 / d, 1 - 15 / d, 20)
    labels = []
    for t in ts:
        x, y = p0 + (p1 - p0) * t
        xi, yi = int(round(x)), int(round(y))
        win = im[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4].reshape(-1, 3)
        dwin = dark[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        ywin = yellow[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        if ywin.any():
            labels.append("undecided")
        elif dwin.any():
            labels.append("line")
        else:
            grayish = np.all((win > 160) & (win < 196), axis=1)
            labels.append("excluded" if grayish.sum() >= 3 else None)
    vals, counts = np.unique([l for l in labels if l], return_counts=True)
    if len(vals) == 0:
        return None
    best = vals[np.argmax(counts)]
    if counts.max() < 0.9 * len(ts):
        return None
    return str(best)


def _find_edges(im, dark, yellow, dots, L):
    n = len(dots)
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = dots[j] - dots[i]
            d = np.hypot(*v)
            if not (0.8 * L < d < 1.2 * L):
                continue
            blocked = False  # no third dot on the segment
            for k in range(n):
                if k in (i, j):
                    continue
                w = dots[k] - dots[i]
                t = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)
                if np.hypot(*(w - t * v)) < 10:
                    blocked = True
                    break
            if blocked:
                continue
            s = _classify_segment(im, dark, yellow, dots[i], dots[j])
            if s:
                edges[(i, j)] = s
    if len(edges) < 30:
        raise ExtractionError("Couldn't trace the grid lines in this screenshot.")
    return edges


def _find_clues(im, dark, dots, L, faces, face_of_point):
    tm = np.load(TEMPLATES)
    n, _, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    clues = {}
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (0.30 * L < h < 0.50 * L and 0.08 * L < w < 0.35 * L and a < 0.12 * L * L):
            continue
        cx, cy = x + w / 2, y + h / 2
        if np.hypot(*(dots - (cx, cy)).T).min() < 0.25 * L:
            continue  # too close to a dot: not a digit
        p = cv2.resize(dark[y:y + h, x:x + w].astype(np.float32), (16, 24))
        val = int(min(tm, key=lambda k2: float(np.sum((p - tm[k2]) ** 2))))
        fi = face_of_point(cx, cy)
        if fi is None:
            raise ExtractionError("Read a clue digit outside any grid cell.")
        if fi in clues:
            raise ExtractionError("Two digits landed in one cell; extraction failed.")
        clues[fi] = val
    if not clues:
        raise ExtractionError("Couldn't read any number clues.")
    return clues


def _faces(dots, edge_list):
    adj = {i: [] for i in range(len(dots))}
    for i, j in edge_list:
        adj[i].append(j)
        adj[j].append(i)

    def ang(a, b):
        d = dots[b] - dots[a]
        return math.atan2(d[1], d[0])

    nxt = {}
    for v in adj:
        nb = sorted(adj[v], key=lambda u: ang(v, u))
        for k, u in enumerate(nb):
            nxt[(u, v)] = (v, nb[(k - 1) % len(nb)])
    faces, seen = [], set()
    for e in edge_list:
        for he in [e, (e[1], e[0])]:
            if he in seen:
                continue
            face, cur = [], he
            while cur not in seen:
                seen.add(cur)
                face.append(cur[0])
                cur = nxt[cur]
            faces.append(face)
    return faces


def _point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > py) != (y2 > py):
            xin = x1 + (py - y1) / (y2 - y1) * (x2 - x1)
            if px < xin:
                inside = not inside
    return inside


def extract(im):
    dark, yellow, _ = _masks(im)
    dots = _find_dots(dark)
    L = _edge_scale(dots)
    edges = _find_edges(im, dark, yellow, dots, L)
    edge_list = sorted(edges)
    faces = _faces(dots, edge_list)
    inner = [f for f in faces if len(f) <= 8]

    def face_of_point(px, py):
        for fi, f in enumerate(faces):
            if len(f) <= 8 and _point_in_poly(px, py, dots[f]):
                return fi
        return None

    clues = _find_clues(im, dark, dots, L, faces, face_of_point)
    if len(inner) != len(faces) - 1:
        raise ExtractionError("Grid didn't parse as a clean tiling; check the screenshot.")
    return dots, edges, faces, clues, L


# ------------------------------------------------------------------- solving

def _face_edges(face, edges):
    out = []
    for k in range(len(face)):
        a, b = face[k], face[(k + 1) % len(face)]
        out.append((a, b) if (a, b) in edges else (b, a))
    return out


def solve(dots, edges, faces, clues):
    edge_list = sorted(edges)
    pool = IDPool()
    ev = {e: pool.id(("e", e)) for e in edge_list}
    cnf = []
    ved = {}
    for e in edge_list:
        ved.setdefault(e[0], []).append(ev[e])
        ved.setdefault(e[1], []).append(ev[e])
    for lits in ved.values():
        for a, b, c in itertools.combinations(lits, 3):
            cnf.append([-a, -b, -c])
        for l in lits:
            cnf.append([-l] + [o for o in lits if o != l])
    for fi, val in clues.items():
        lits = [ev[e] for e in _face_edges(faces[fi], edges)]
        if val == 0:
            cnf += [[-l] for l in lits]
        else:
            cnf += CardEnc.equals(lits, val, vpool=pool,
                                  encoding=EncType.seqcounter).clauses

    def comps(sel):
        ad = {}
        for a, b in sel:
            ad.setdefault(a, []).append(b)
            ad.setdefault(b, []).append(a)
        seen, out = set(), []
        for v in ad:
            if v in seen:
                continue
            stack, comp = [v], set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(ad[u])
            seen |= comp
            out.append(comp)
        return out

    s = Cadical195(bootstrap_with=cnf)
    for _ in range(200):
        if not s.solve():
            raise ExtractionError(
                "The clues I read have no valid loop - I probably misread the grid.")
        m = set(l for l in s.get_model() if l > 0)
        sel = [e for e in edge_list if ev[e] in m]
        cc = comps(sel)
        if len(cc) <= 1:
            return set(sel)
        cc.sort(key=len)
        for comp in cc[:-1]:
            s.add_clause([-ev[e] for e in sel if e[0] in comp and e[1] in comp])
    raise ExtractionError("Solver did not converge.")


# ---------------------------------------------------------------- hint logic

def _propagate(state, adj, face_data):
    """One state dict -> propagate simple rules. Returns None on contradiction."""
    changed = True
    while changed:
        changed = False
        for es in adj.values():
            lines = [e for e in es if state[e] == "line"]
            und = [e for e in es if state[e] == "undecided"]
            if len(lines) > 2 or (len(lines) == 1 and not und):
                return None
            if len(lines) == 2 and und:
                for e in und:
                    state[e] = "excluded"
                changed = True
            elif len(lines) == 1 and len(und) == 1:
                state[und[0]] = "line"
                changed = True
            elif not lines and len(und) == 1:
                state[und[0]] = "excluded"
                changed = True
        for val, es in face_data:
            lines = sum(state[e] == "line" for e in es)
            und = [e for e in es if state[e] == "undecided"]
            if lines > val or lines + len(und) < val:
                return None
            if lines == val and und:
                for e in und:
                    state[e] = "excluded"
                changed = True
            elif lines + len(und) == val and und:
                for e in und:
                    state[e] = "line"
                changed = True
    return state


def find_hint(dots, edges, faces, clues, state):
    adj = {}
    for e in edges:
        adj.setdefault(e[0], []).append(e)
        adj.setdefault(e[1], []).append(e)
    face_data = [(val, _face_edges(faces[fi], edges)) for fi, val in clues.items()]
    face_center = {fi: dots[faces[fi]].mean(axis=0) for fi in clues}

    # level 1: single-cell / single-dot rules
    for fi, val in clues.items():
        es = _face_edges(faces[fi], edges)
        lines = sum(state[e] == "line" for e in es)
        und = [e for e in es if state[e] == "undecided"]
        c = face_center[fi]
        if val == 0 and und:
            return ("X", und, f"A cell marked 0 has no loop edges at all - "
                    f"cross out every edge of the 0 cell.", c)
        if lines == val and und:
            return ("X", und, f"The {val} cell here already has {val} line"
                    f"{'s' if val != 1 else ''} - cross out its remaining edges.", c)
        if und and lines + len(und) == val:
            return ("LINE", und, f"The {val} cell here has only {val} usable edges "
                    f"left - all of them must be lines.", c)
    for v, es in adj.items():
        lines = [e for e in es if state[e] == "line"]
        und = [e for e in es if state[e] == "undecided"]
        if len(lines) == 2 and und:
            return ("X", und, "This dot already has two lines through it - "
                    "cross out its other edges.", dots[v])
        if len(lines) == 1 and len(und) == 1:
            return ("LINE", und, "A line ends at this dot with only one way to "
                    "continue - extend it.", dots[v])
        if not lines and len(und) == 1:
            return ("X", und, "This edge would be a dead end - a line here could "
                    "never continue. Cross it out.", dots[v])

    # level 2: one-edge what-if
    for e in [e for e in edges if state[e] == "undecided"]:
        for guess, verdict in [("line", "X"), ("excluded", "LINE")]:
            st = dict(state)
            st[e] = guess
            if _propagate(st, adj, face_data) is None:
                word = ("part of the loop" if guess == "line" else "crossed out")
                fix = ("cross it out" if verdict == "X" else "draw it in")
                mid = (dots[e[0]] + dots[e[1]]) / 2
                return (verdict, [e],
                        f"Try a what-if on the highlighted edge: assume it is "
                        f"{word} and follow the forced moves through the nearby "
                        f"clues - you'll hit a contradiction. So {fix}.", mid)
    return None


# ----------------------------------------------------------------- top level

def annotate(im, dots, hint_edges, verdict):
    out = im.copy()
    color = (0, 0, 255) if verdict == "X" else (255, 100, 0)
    for a, b in hint_edges:
        p0 = tuple(np.round(dots[a]).astype(int))
        p1 = tuple(np.round(dots[b]).astype(int))
        cv2.line(out, p0, p1, (255, 255, 255), 13, cv2.LINE_AA)
        cv2.line(out, p0, p1, color, 7, cv2.LINE_AA)
    return out


def hint_from_image(im):
    """Main entry. Returns (message, annotated_bgr_image)."""
    # Normalize to the calibrated working width; detection constants assume it.
    target_w = 922
    if im.shape[1] != target_w:
        scale = target_w / im.shape[1]
        im = cv2.resize(im, (target_w, int(im.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    dots, edges, faces, clues, L = extract(im)
    solution = solve(dots, edges, faces, clues)

    wrong_lines = [e for e, s in edges.items() if s == "line" and e not in solution]
    wrong_x = [e for e, s in edges.items() if s == "excluded" and e in solution]
    if wrong_lines or wrong_x:
        bad = wrong_lines + wrong_x
        out = annotate(im, dots, bad, "X")
        parts = []
        if wrong_lines:
            parts.append(f"{len(wrong_lines)} drawn line"
                         f"{'s' if len(wrong_lines) > 1 else ''} that shouldn't be there")
        if wrong_x:
            parts.append(f"{len(wrong_x)} crossed-out edge"
                         f"{'s' if len(wrong_x) > 1 else ''} the loop actually needs")
        return ("Before any hint - there "
                f"{'are' if len(bad) > 1 else 'is'} {' and '.join(parts)}. "
                "They're highlighted in red. Fix those first.", out)

    n_lines = sum(1 for s in edges.values() if s == "line")
    if n_lines == len(solution) and not wrong_lines:
        und = [e for e, s in edges.items() if s == "undecided"]
        if not any(e in solution for e in und):
            return ("The loop is complete - puzzle solved. Nothing left to hint!",
                    im.copy())

    h = find_hint(dots, edges, faces, clues, edges)
    if h is None:
        # fall back: reveal one correct edge near existing progress
        e = next(e for e in sorted(edges) if edges[e] == "undecided" and e in solution)
        out = annotate(im, dots, [e], "LINE")
        return ("No short logical step found - here's a freebie: the highlighted "
                "edge is part of the loop.", out)
    verdict, hes, msg, _ = h
    action = "Cross out" if verdict == "X" else "Draw in"
    return (f"No mistakes so far. {msg} ({action} the highlighted edge"
            f"{'s' if len(hes) > 1 else ''}.)", annotate(im, dots, hes, verdict))
