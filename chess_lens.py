"""
Light-blue chess overlay: finds the board, calibrates itself, detects your side and draws the best-move arrow.

Keys:
  F8 - toggles modes:
       * SETUP: move/resize the frame over the board (right-click the frame - exit)
       * PLAY:  frame hidden, clicks pass through the window, the script follows the game
  F2 - automatically find the chessboard on screen and put the frame on it

What happens automatically
  * Board search (F2): finds the 8x8 grid by square corners - independent of theme and site.
  * Auto-calibration: when the starting position is on the board, the script learns what
    your pieces look like and checks itself (all 32 pieces must be recognised correctly).
  * Side: whoever is at the bottom is the side the script plays for.
  * Reliability: silhouette matching with shift tolerance, a "margin" check against the
    second-best candidate, filtering of the site's move-hint dots, a position is accepted
    only after several identical frames, move tracking (including two moves at once).

On start the script asks for administrator rights itself (UAC prompt).

Install:  pip install PyQt5 python-chess mss opencv-python numpy
"""
import sys
import os
import json
import glob
import math
import itertools
import queue
import ctypes

import numpy as np
import cv2
import mss
import chess
import chess.engine

from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygonF, QFont, QFontMetrics
from PyQt5.QtWidgets import QApplication, QWidget

# ----------------------------- SETTINGS -----------------------------
ENGINE_PATH = ""            # empty = auto-find the engine next to the script / .exe (see find_engine); or a full path
ENGINE_ARGS = []            # e.g. ["--weights=D:\\Projects\\chess\\lc0\\network.pb.gz"]
MOVE_TIME = 0.8             # seconds per move
# next to the script / next to the .exe (in a --onefile build __file__ points to a temp folder
# that is deleted on exit, so settings and templates are stored next to the exe)
HERE = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(HERE, "pieces_templates.npz")
SETTINGS_FILE = os.path.join(HERE, "overlay_settings.json")
SCAN_MS = 350               # how often to scan the screen (ms)
TPL = 32                    # silhouette template size
FEATURE_VERSION = 3         # bump when the silhouette pipeline changes (older saved templates are ignored)
SHIFT = 2                   # silhouette shift tolerance (in template pixels)
EMPTY_AREA = 0.05           # share of non-background pixels below which a square counts as empty
DIFF_THRESH = 50            # difference-from-background threshold (sum over BGR)
MIN_IOU = 0.40              # minimum silhouette/template similarity
MIN_MARGIN = 0.08           # below this gap two candidates count as ambiguous (resolved by game rules)
REBUILD_FRAMES = 4          # identical frames needed to accept a position "from scratch"
BAD_READS_LIMIT = 12        # consecutive bad frames before the F2 hint appears
TEXTURE_ON = 40.0           # board noise above this -> textured-board mode
TEXTURE_OFF = 20.0          # board noise below this -> flat-board mode (in between: keep the current mode)
WAIT_DUMP_TICKS = 15        # ticks of waiting for the starting position before debug files are saved
BLUE = (64, 196, 255)       # overlay light-blue colour (R, G, B)
# ---------------------------------------------------------------------

def find_engine():
    """Locate the UCI engine. An explicit ENGINE_PATH wins; otherwise look in the folder of the
    script / .exe (and one level of sub-folders) for lc0 first, then Stockfish."""
    if ENGINE_PATH:
        return ENGINE_PATH if os.path.isfile(ENGINE_PATH) else None
    patterns = ["lc0.exe", "lc0", os.path.join("*", "lc0.exe"), os.path.join("*", "lc0"),
                "stockfish*.exe", "stockfish*", os.path.join("*", "stockfish*.exe"),
                os.path.join("*", "stockfish*")]
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(HERE, pat))):
            if os.path.isfile(path) and not path.lower().endswith((".txt", ".md", ".zip", ".gz", ".pb")):
                return path
    return None


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
IS_WIN = sys.platform == "win32"
user32 = ctypes.windll.user32 if IS_WIN else None
VK_F8, VK_F2 = 0x77, 0x71
BAD_TEXT = "Can't read the board. Press F2 to find the board again"   # may get a suffix
ERROR_WORDS = ("error", "failed", "not found", "can't read")


def blue(alpha=255):
    return QColor(BLUE[0], BLUE[1], BLUE[2], alpha)


# ----------------------------- geometry -----------------------------
def rc_to_square(r, c, white_bottom):
    return chess.square(c, 7 - r) if white_bottom else chess.square(7 - c, r)


def square_to_rc(sq, white_bottom):
    f, rk = chess.square_file(sq), chess.square_rank(sq)
    return (7 - rk, f) if white_bottom else (rk, 7 - f)


def split_squares(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    e = [int(round(i * side / 8)) for i in range(9)]
    return [[frame[e[r]:e[r + 1], e[c]:e[c + 1]] for c in range(8)] for r in range(8)]


# ----------------------------- board detection (F2) -----------------------------
def detect_board(gray_u8):
    """Finds an 8x8 chessboard grid in a screenshot (grayscale image).

    Idea: at square corners the brightness forms a "saddle" (light-dark / dark-light).
    For every square size s we compute the saddle response at every point, then sum
    the responses over the 7x7 lattice of inner corners with alternating sign (-1)^(i+j) -
    a "phase" check that ordinary noise and unrelated images do not pass.
    Fast: the square size is searched on a half-size copy, then refined (+-2 px)
    at working resolution. Returns (x, y, side, score) in image pixels, or None.
    """
    H, W = gray_u8.shape[:2]
    f = 0.5 if max(H, W) > 2600 else 1.0                  # working resolution
    g1 = gray_u8 if f == 1.0 else cv2.resize(gray_u8, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    g1 = g1.astype(np.float32)
    gc = cv2.resize(g1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def compute(g, s):
        h, w = g.shape
        wv, hv = w - 6 * s, h - 6 * s
        if wv <= 0 or hv <= 0:
            return None
        q = max(3, int(s * 0.3))
        M = cv2.boxFilter(g, -1, (q, q), anchor=(0, 0), normalize=True,
                          borderType=cv2.BORDER_REPLICATE)
        R = M.copy()                    # BR
        R[q:, q:] += M[:-q, :-q]        # TL
        R[q:, :] -= M[:-q, :]           # TR
        R[:, q:] -= M[:, :-q]           # BL
        A = R[:, 0:wv].copy()
        for i in range(1, 7):
            if i % 2:
                A -= R[:, i * s:i * s + wv]
            else:
                A += R[:, i * s:i * s + wv]
        L = A[0:hv, :].copy()
        for j in range(1, 7):
            if j % 2:
                L -= A[j * s:j * s + hv, :]
            else:
                L += A[j * s:j * s + hv, :]
        idx = np.unravel_index(int(np.argmax(np.abs(L))), L.shape)
        val = float(L[idx])
        return abs(val), int(idx[1]), int(idx[0]), val, R

    # 1) coarse square-size search on a half-size copy
    hc, wc = gc.shape
    smin_c, smax_c = max(7, int(12 * f)), int(min(hc, wc) / 8)
    best_c, best_sc = None, 0.0
    for s in range(smin_c, smax_c + 1):
        res = compute(gc, s)
        if res and res[0] > best_sc:
            best_sc, best_c = res[0], s
    if best_c is None or best_sc / 49.0 < 4:              # nothing on screen looks like a board
        return None

    # 2) refinement at working resolution
    h1, w1 = g1.shape
    smax = int(min(h1, w1) / 8)
    best = None
    for s in range(2 * best_c - 2, 2 * best_c + 3):
        if s < 14 or s > smax:
            continue
        res = compute(g1, s)
        if res and (best is None or res[0] > best[0][0]):
            best = (res, s)
    if best is None:
        return None
    (sc, x0, y0, val, R), best_s = best

    sign = 1.0 if val > 0 else -1.0
    mean_c = sc / 49.0
    contrib = np.array([[((-1) ** (i + j)) * sign * R[y0 + j * best_s, x0 + i * best_s]
                         for i in range(7)] for j in range(7)])
    frac = float(np.mean(contrib > 0.25 * mean_c))
    if mean_c < 8 or frac < 0.8:                          # not enough contrast / incomplete grid
        return None

    x = max(0, int(round((x0 - best_s) / f)))
    y = max(0, int(round((y0 - best_s) / f)))
    side = int(round(8 * best_s / f))
    return x, y, side, sc


# ----------------------------- square features -----------------------------
def ring_noise(sq):
    """Texture level of a square: spread of the colours along its (normally empty) outer ring."""
    h, w = sq.shape[:2]
    m = max(2, int(min(h, w) * 0.12))
    ring = np.concatenate([
        sq[:m].reshape(-1, 3), sq[-m:].reshape(-1, 3),
        sq[:, :m].reshape(-1, 3), sq[:, -m:].reshape(-1, 3),
    ]).astype(np.int16)
    bg = np.median(ring, axis=0).astype(np.int16)
    return float(np.percentile(np.abs(ring - bg).sum(axis=1), 90))


def extract_mask(sq, textured=False):
    """Solid piece silhouette of one square. Returns (mask uint8, region BGR, inner-area normaliser).

    * background colour = median of the square's outer ring (so highlights don't matter)
    * a thin outline is enough: white pieces on white squares differ from the square ONLY by their outline,
      so gaps in the outline are closed and the inside is filled
    * `textured` (wood-grain style boards): smooth the grain and raise the threshold to the measured noise"""
    h, w = sq.shape[:2]
    if textured:
        sq = cv2.GaussianBlur(sq, (0, 0), max(0.8, min(h, w) / 48.0))
    m = max(2, int(min(h, w) * 0.12))                   # ring used to estimate the background
    ring = np.concatenate([
        sq[:m].reshape(-1, 3), sq[-m:].reshape(-1, 3),
        sq[:, :m].reshape(-1, 3), sq[:, -m:].reshape(-1, 3),
    ])
    bg = np.median(ring, axis=0).astype(np.int16)       # square background colour (incl. highlights)
    thr = float(DIFF_THRESH)
    if textured:
        ring_diff = np.abs(ring.astype(np.int16) - bg).sum(axis=1)
        thr = min(160.0, max(thr, 1.5 * float(np.percentile(ring_diff, 90)) + 12.0))
    ms = max(1, int(min(h, w) * 0.04))                  # piece bases come close to the square's edge
    region = sq[ms:h - ms, ms:w - ms]
    diff = np.abs(region.astype(np.int16) - bg).sum(axis=2)
    mask = (diff > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # close gaps in the outline
    # keep only the big blob(s): a piece is one solid shape, speckle/labels are small ones
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.nonzero(areas >= max(6, 0.15 * areas.max()))[0] + 1
        mask = np.isin(labels, keep).astype(np.uint8)
        # fill the inside of the outline: flood the background from outside, whatever is left is the piece
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(padded, flood_mask, (0, 0), 2)           # background becomes 2, holes stay 0
        mask = (padded[1:-1, 1:-1] != 2).astype(np.uint8)
    else:
        mask = np.zeros_like(mask)
    return mask, region, float((h - 2 * m) * (w - 2 * m))


def square_features(sq, textured=False):
    """Piece silhouette (float TPLxTPL, 0..1), occupied-pixel share, piece brightness."""
    mask, region, inner_area = extract_mask(sq, textured)
    area = float(mask.sum()) / inner_area
    small = cv2.resize(mask.astype(np.float32), (TPL, TPL), interpolation=cv2.INTER_AREA)
    bright = None
    if mask.sum() > 0:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
        sel = core.astype(bool) if core.sum() >= 15 else mask.astype(bool)
        bright = float(np.median(gray[sel]))             # median is robust to the piece outline
    return small, area, bright


def looks_like_hint(small):
    """A legal-move hint dot (small centred circle), not a piece."""
    m = small > 0.5
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return True
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    return bw <= 0.5 * TPL and bh <= 0.5 * TPL and m.sum() / float(bw * bh) > 0.55


def match_scores(small, tpl):
    """Similarity of a silhouette to every template, tolerating small shifts."""
    sp = np.pad(small, SHIFT)
    wins = np.lib.stride_tricks.sliding_window_view(sp, (TPL, TPL)).reshape(-1, TPL, TPL)
    inter = np.minimum(wins[None], tpl[:, None]).sum(axis=(2, 3))
    union = wins.sum(axis=(1, 2))[None, :] + tpl.sum(axis=(1, 2))[:, None] - inter   # |A|+|B|-|A&B|
    return (inter / np.maximum(union, 1e-6)).max(axis=1)


def best_match(small, tpl):
    """Returns (type index, similarity, gap to the second-best candidate)."""
    sims = match_scores(small, tpl)
    order = np.argsort(sims)[::-1]
    return int(order[0]), float(sims[order[0]]), float(sims[order[0]] - sims[order[1]])


def start_diagnosis(feats):
    """(pieces seen in the 4 home rows out of 32, stray marks in the 4 middle rows)."""
    home = sum(1 for r in (0, 1, 6, 7) for c in range(8) if feats[r][c][1] >= EMPTY_AREA)
    stray = sum(1 for r in (2, 3, 4, 5) for c in range(8) if feats[r][c][1] >= EMPTY_AREA)
    return home, stray


def start_orientation(feats, why=None):
    """If the board shows the starting position: True (White at the bottom) / False (Black at the bottom), else None.
    Works without templates: square occupancy, piece brightness, pawn similarity.
    `why` (list) receives the reason when the occupancy looks right but another check fails."""
    for r in range(8):
        for c in range(8):
            occupied = feats[r][c][1] >= EMPTY_AREA
            if occupied != (r in (0, 1, 6, 7)):
                return None
    top = np.mean([feats[r][c][2] or 0 for r in (0, 1) for c in range(8)])
    bottom = np.mean([feats[r][c][2] or 0 for r in (6, 7) for c in range(8)])
    if abs(top - bottom) < 40:
        if why is not None:
            why.append(f"White/Black pieces look equally bright (gap {abs(top - bottom):.0f} < 40)")
        return None
    pawns = np.stack([feats[r][c][0] for r in (1, 6) for c in range(8)])
    mean = pawns.mean(axis=0)
    worst = 1.0
    for pw in pawns:                                     # all 16 pawns must look alike
        union = np.maximum(pw, mean).sum()
        worst = min(worst, np.minimum(pw, mean).sum() / union if union else 0.0)
    if worst < 0.6:
        if why is not None:
            why.append(f"pawns don't look alike (min similarity {worst:.2f} < 0.6)")
        return None
    return bool(bottom > top)


def build_templates(feats, white_bottom, why=None):
    """Piece templates from the starting position + self-check. Returns (tpl, thr) or None."""
    start = chess.Board()
    masks = {pt: [] for pt in PIECE_TYPES}
    br = {chess.WHITE: [], chess.BLACK: []}
    for r in range(8):
        for c in range(8):
            p = start.piece_at(rc_to_square(r, c, white_bottom))
            if not p:
                continue
            small, area, bright = feats[r][c]
            masks[p.piece_type].append(small)
            br[p.color].append(128.0 if bright is None else bright)
    tpl = np.stack([np.mean(masks[pt], axis=0) for pt in PIECE_TYPES]).astype(np.float32)
    if any((t > 0.5).sum() < 20 for t in tpl):
        if why is not None:
            why.append("a piece template came out too small")
        return None
    thr = float((np.mean(br[chess.WHITE]) + np.mean(br[chess.BLACK])) / 2)
    for r in range(8):                                   # self-check: all 32 pieces must be recognised correctly
        for c in range(8):
            p = start.piece_at(rc_to_square(r, c, white_bottom))
            if not p:
                continue
            small, area, bright = feats[r][c]
            i, sim, margin = best_match(small, tpl)
            name = chess.square_name(rc_to_square(r, c, white_bottom))
            if PIECE_TYPES[i] != p.piece_type or sim < MIN_IOU:
                if why is not None:
                    why.append(f"self-check: {name} is a {chess.piece_name(p.piece_type)} but read as "
                               f"{chess.piece_name(PIECE_TYPES[i])} (similarity {sim:.2f})")
                return None
            if ((bright or 0) > thr) != (p.color == chess.WHITE):
                if why is not None:
                    why.append(f"self-check: wrong piece colour at {name}")
                return None
    return tpl, thr


def read_grid(feats, tpl, thr, prior=None, why=None, alts=None):
    """{(row, column): (type, colour)} or None if the frame can't be read reliably.
    `prior` {(row, column): piece type} = what we already expect there; it only decides near-ties
    between look-alike pieces. `why` (list) receives the first problem found; `alts` (list) receives
    ((row, column), second-best piece type) for every ambiguous square."""
    grid = {}
    for r in range(8):
        for c in range(8):
            small, area, bright = feats[r][c]
            # a hint dot covers at most ~25% of a square - no need to check large silhouettes
            if area < EMPTY_AREA or (area < 0.27 and looks_like_hint(small)):
                continue
            sims = match_scores(small, tpl)
            order = np.argsort(sims)[::-1]
            i = int(order[0])
            sim = float(sims[i])
            if sim < MIN_IOU:
                if why is not None:
                    why.append(f"row {r} col {c}: best similarity {sim:.2f} < {MIN_IOU}")
                return None                              # animation/interference frame - skip it
            if sim - float(sims[order[1]]) < MIN_MARGIN:
                j2 = int(order[1])
                if prior:
                    expected = prior.get((r, c))         # near-tie: prefer the piece we expect here
                    if PIECE_TYPES[j2] == expected:
                        i, j2 = j2, int(order[0])
                if alts is not None:
                    alts.append(((r, c), PIECE_TYPES[j2]))
            color = chess.WHITE if (bright or 0) > thr else chess.BLACK
            grid[(r, c)] = (PIECE_TYPES[i], color)
    return grid


def debug_report(feats, tpl, thr, header):
    """Plain-text dump of what the recognizer sees (saved next to the app when reading fails)."""
    lines = [header, ""]
    home, stray = start_diagnosis(feats)
    lines.append(f"start-position check: pieces seen {home}/32, stray marks {stray}")
    why = []
    lines.append(f"start_orientation: {start_orientation(feats, why)} {why}")
    lines.append(f"settings: DIFF_THRESH={DIFF_THRESH} EMPTY_AREA={EMPTY_AREA} MIN_IOU={MIN_IOU} "
                 f"MIN_MARGIN={MIN_MARGIN} templates={'yes' if tpl is not None else 'no'} colour_thr={thr:.0f}")
    lines.append("")
    lines.append("row col  area  brightness  best-type similarity margin")
    for r in range(8):
        for c in range(8):
            small, area, bright = feats[r][c]
            extra = ""
            if tpl is not None and area >= EMPTY_AREA:
                i, sim, margin = best_match(small, tpl)
                extra = f"  {chess.piece_name(PIECE_TYPES[i]):7s} {sim:.2f} {margin:.2f}"
            lines.append(f"{r:3d} {c:3d}  {area:.3f}  {'-' if bright is None else format(bright, '.0f'):>6}{extra}")
    return "\n".join(lines)


def guess_orientation(grid):
    """Mid-game: White pieces are on average closer to the bottom of the screen -> White at the bottom."""
    w = [r for (r, c), (pt, col) in grid.items() if col == chess.WHITE]
    b = [r for (r, c), (pt, col) in grid.items() if col == chess.BLACK]
    if not w or not b:
        return True
    return bool(np.mean(w) >= np.mean(b))


def bb_key(b):
    """Piece placement as a tuple of bitboards (compares much faster than a FEN string)."""
    return (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])


def placement_for(grid, white_bottom):
    b = chess.Board(None)
    for (r, c), (pt, col) in grid.items():
        b.set_piece_at(rc_to_square(r, c, white_bottom), chess.Piece(pt, col))
    if len(b.pieces(chess.KING, chess.WHITE)) != 1 or len(b.pieces(chess.KING, chess.BLACK)) != 1:
        return None
    return b.board_fen()


# ----------------------------- engine thread -----------------------------
class EngineWorker(QThread):
    result = pyqtSignal(str, str)   # fen, uci
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.q = queue.Queue()

    def submit(self, fen):
        self.q.put(fen)

    def stop(self):
        self.q.put(None)

    def run(self):
        engine_path = find_engine()
        if not engine_path:
            self.status.emit("Engine not found: put lc0.exe (with its network file) next to the app")
            return
        try:
            extra = {}
            if IS_WIN:
                extra["creationflags"] = 0x08000000     # CREATE_NO_WINDOW: don't open the engine console
            extra["cwd"] = os.path.dirname(os.path.abspath(engine_path))   # engine finds its network file there
            engine = chess.engine.SimpleEngine.popen_uci([engine_path] + ENGINE_ARGS, **extra)
        except Exception as e:
            self.status.emit(f"Engine failed to start: {e}")
            return
        try:
            while True:
                fen = self.q.get()
                if fen is None:
                    break
                stop = False
                while True:                      # take only the newest position
                    try:
                        nxt = self.q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        stop = True
                        break
                    fen = nxt
                if stop:
                    break
                try:
                    board = chess.Board(fen)
                    res = engine.play(board, chess.engine.Limit(time=MOVE_TIME))
                    if res.move:
                        self.result.emit(fen, res.move.uci())
                except Exception as e:
                    self.status.emit(f"Engine error: {e}")
        finally:
            try:
                engine.quit()
            except Exception:
                pass


# ----------------------------- overlay -----------------------------
class ChessOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setGeometry(500, 200, 640, 640)
        self.load_geometry()

        self.sct = mss.mss()
        self.edit_mode = True
        self.busy = False               # board search (F2) in progress
        self.white_bottom = True
        self.my_color = chess.WHITE
        self.textured = False           # textured (wood-grain) board mode
        self.tpl = None
        self.thr = 128.0
        self.status_text = ""
        self.play_notice = ""
        self.flash_text = ""
        self.flash_error = False
        self._flash_id = 0
        self.dragging = False
        self.resizing = False
        self.drag_pos = QPoint()
        self._keys = {}

        self.reset_state()
        self.load_templates()

        self.worker = EngineWorker()
        self.worker.result.connect(self.on_engine_result)
        self.worker.status.connect(lambda t: self.set_status(t, notice=True))
        self.worker.start()

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.tick)
        self.scan_timer.start(SCAN_MS)

        self.key_timer = QTimer(self)
        self.key_timer.timeout.connect(self.poll_keys)
        self.key_timer.start(50)

    # ---- saving frame position and size ----
    def load_geometry(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            x, y, side = int(d["x"]), int(d["y"]), max(200, int(d["size"]))
            center = QPoint(x + side // 2, y + side // 2)
            if any(scr.geometry().contains(center) for scr in QApplication.screens()):
                self.setGeometry(x, y, side, side)
        except Exception:
            pass

    def save_geometry(self):
        try:
            g = self.geometry()
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({"x": g.x(), "y": g.y(), "size": min(g.width(), g.height())}, f)
        except Exception:
            pass

    # ---- state ----
    def reset_state(self):
        self.board = chess.Board()
        self.last_placement = None
        self.candidate = None
        self.stable = 0
        self.arrow = None
        self.locked = False             # side already determined in this game
        self.calibrated_game = False    # auto-calibration already done in this game
        self.bad_reads = 0
        # scan cache
        self._feats = None
        self._prev_small = None
        self._prev_side = 0
        self._done_feats = None         # frame that has already been fully processed
        self._failed_feats = None       # frame on which calibration failed the self-check
        self._grid_feats = None
        self._grid = None
        self._alts = []
        self._noise = 0.0
        self._why = ""                  # first reason the last frame couldn't be read
        self._last_frame = None
        self.wait_ticks = 0             # ticks spent waiting for the starting position
        self._dump_done = False         # debug files already saved in this session

    def set_orientation(self, white_bottom):
        self.white_bottom = white_bottom
        self.my_color = chess.WHITE if white_bottom else chess.BLACK

    def set_status(self, text, notice=False):
        if text and text != self.status_text:
            print(text)
        self.status_text = text
        self.play_notice = text if notice else ""
        self.update()

    def flash(self, text, ms=2500, error=False):
        """Temporary centred message."""
        self._flash_id += 1
        fid = self._flash_id
        self.flash_text, self.flash_error = text, error
        if ms > 0:
            QTimer.singleShot(ms, lambda: self._clear_flash(fid))
        self.update()

    def _clear_flash(self, fid):
        if fid == self._flash_id:
            self.flash_text = ""
            self.update()

    # ---- Windows specifics ----
    def apply_win_flags(self):
        """Exclude the window from screen capture and toggle click-through."""
        if not IS_WIN:
            return
        hwnd = int(self.winId())
        try:
            user32.SetWindowDisplayAffinity(hwnd, 0x11)  # WDA_EXCLUDEFROMCAPTURE (Win10 2004+)
        except Exception:
            pass
        style = user32.GetWindowLongW(hwnd, -20)
        style |= 0x80000
        style = (style | 0x20) if not self.edit_mode else (style & ~0x20)
        user32.SetWindowLongW(hwnd, -20, style)

    def key_edge(self, vk):
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was = self._keys.get(vk, False)
        self._keys[vk] = down
        return down and not was

    def poll_keys(self):
        if not IS_WIN:
            return
        if self.key_edge(VK_F8):
            self.toggle_mode()
        if self.key_edge(VK_F2):
            self.find_board()

    def toggle_mode(self):
        self.edit_mode = not self.edit_mode
        self.apply_win_flags()
        self.reset_state()
        if self.edit_mode:
            self.set_status("")
        else:
            self.set_status("Waiting for the starting position..." if self.tpl is None else "",
                            notice=self.tpl is None)
        self.update()

    # ---- automatic board detection (F2) ----
    def find_board(self):
        if self.busy:
            return
        self.busy = True
        try:
            self.hide()                                   # so the frame doesn't end up in the screenshot
            QApplication.processEvents()
            QThread.msleep(150)
            shots = []
            for mon in self.sct.monitors[1:]:
                shot = self.sct.grab(mon)
                img = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)
                shots.append((mon, cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)))
            self.show()
            self.flash("Looking for the board...", 0)
            QApplication.processEvents()

            best = None
            for mon, gray in shots:
                res = detect_board(gray)
                if res and (best is None or res[3] > best[3]):
                    best = (mon["left"] + res[0], mon["top"] + res[1], res[2], res[3])
            if best is None:
                self.flash("Board not found. Align the frame manually (F8)", 3500, error=True)
                return
            x, y, side, _ = best
            self.setGeometry(x, y, side, side)
            self.save_geometry()
            self.reset_state()
            self.flash(f"Board found ({side}x{side})", 2000)
        except Exception as e:
            self.show()
            self.flash(f"Board search error: {e}", 3500, error=True)
        finally:
            self.busy = False

    # ---- capture ----
    def grab(self):
        g = self.geometry()
        mon = {"top": g.y(), "left": g.x(), "width": g.width(), "height": g.height()}
        shot = self.sct.grab(mon)
        img = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)   # zero-copy
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def scan(self):
        """Features of all squares. If nothing on the board changed, returns the same object
        (so we don't recompute the same thing every 350 ms)."""
        frame = self.grab()
        self._last_frame = frame
        side = min(frame.shape[:2])
        small = cv2.resize(cv2.cvtColor(frame[:side, :side], cv2.COLOR_BGR2GRAY), (64, 64),
                           interpolation=cv2.INTER_AREA)
        if self._feats is not None and self._prev_small is not None \
                and self._prev_side == side:
            diff = cv2.absdiff(small, self._prev_small).reshape(8, 8, 8, 8).mean(axis=(1, 3))
            if diff.max() < 2.0:                          # no square changed noticeably
                return self._feats
        sq = split_squares(frame)
        noise = float(np.median([ring_noise(sq[r][c]) for r in range(8) for c in range(8)]))
        if noise > TEXTURE_ON:
            self.textured = True
        elif noise < TEXTURE_OFF:
            self.textured = False
        self._noise = noise
        self._feats = [[square_features(sq[r][c], self.textured) for c in range(8)] for r in range(8)]
        self._prev_small, self._prev_side = small, side
        return self._feats

    # ---- templates ----
    def load_templates(self):
        if os.path.exists(TEMPLATE_FILE):
            try:
                d = np.load(TEMPLATE_FILE)
                if int(d["ver"]) == FEATURE_VERSION:
                    self.tpl = d["tpl"].astype(np.float32)
                    self.thr = float(d["thr"])
            except Exception:
                self.tpl = None

    # ---- diagnostics ----
    def prior_map(self):
        """What we expect on each square (screen coordinates) from the tracked game."""
        if not self.locked:
            return None
        return {square_to_rc(sq, self.white_bottom): pc.piece_type
                for sq, pc in self.board.piece_map().items()}

    def dump_debug(self, feats, header):
        """Save the captured board + what the recognizer sees, so a failure can be analysed."""
        self._dump_done = True
        try:
            if self._last_frame is not None:
                ok, buf = cv2.imencode(".png", self._last_frame)
                if ok:
                    with open(os.path.join(HERE, "debug_board.png"), "wb") as f:
                        f.write(buf.tobytes())
            g = self.geometry()
            header += (f"\nframe: x={g.x()} y={g.y()} size={g.width()}x{g.height()} "
                       f"white_bottom={self.white_bottom} locked={self.locked} "
                       f"board_noise={self._noise:.1f} textured_mode={self.textured}")
            with open(os.path.join(HERE, "debug_info.txt"), "w", encoding="utf-8") as f:
                f.write(debug_report(feats, self.tpl, self.thr, header))
        except Exception:
            pass

    # ---- game tracking ----
    def track_move(self, placement):
        """Find a legal move (or two moves in a row) that produces the new placement."""
        if placement == chess.STARTING_BOARD_FEN:
            self.board = chess.Board()
            return True
        tb = chess.Board(None)
        tb.set_board_fen(placement)
        target = bb_key(tb)
        if bb_key(self.board) == target:
            return True
        for mv in list(self.board.legal_moves):
            self.board.push(mv)
            if bb_key(self.board) == target:
                return True
            self.board.pop()
        for mv in list(self.board.legal_moves):           # missed frame: two moves in a row
            self.board.push(mv)
            for mv2 in list(self.board.legal_moves):
                self.board.push(mv2)
                if bb_key(self.board) == target:
                    return True
                self.board.pop()
            self.board.pop()
        return False

    def track_with_alts(self, grid, placement, alts):
        """track_move(), and if no legal move explains the frame, retry with ambiguous squares
        (look-alike pieces) flipped to their second-best reading - the game rules pick the right one."""
        if self.track_move(placement):
            return True
        alts = alts[:4]
        for k in (1, 2):
            for combo in itertools.combinations(alts, k):
                g2 = dict(grid)
                for rc, alt_type in combo:
                    g2[rc] = (alt_type, grid[rc][1])
                alt_placement = placement_for(g2, self.white_bottom)
                if alt_placement and alt_placement != placement and self.track_move(alt_placement):
                    return True
        return False

    def rebuild(self, placement):
        """Rebuild the position from scratch (mid-game / lost track)."""
        for turn in (self.my_color, not self.my_color):
            b = chess.Board(None)
            b.set_board_fen(placement)
            b.turn = turn
            rights = 0
            for color, rank in ((chess.WHITE, 0), (chess.BLACK, 7)):
                if b.piece_at(chess.square(4, rank)) == chess.Piece(chess.KING, color):
                    for file in (0, 7):
                        if b.piece_at(chess.square(file, rank)) == chess.Piece(chess.ROOK, color):
                            rights |= chess.BB_SQUARES[chess.square(file, rank)]
            b.castling_rights = rights
            if b.is_valid():
                self.board = b
                return True
        return False

    def tick(self):
        if self.edit_mode or self.busy:
            return
        try:
            feats = self.scan()
            if feats is self._done_feats:
                return                                    # frame unchanged, everything already processed
            start_why = []
            start = start_orientation(feats, start_why)

            if start is not None:
                # Starting position: auto-calibration + side from whoever is at the bottom
                if not self.calibrated_game and feats is not self._failed_feats:
                    calib_why = []
                    built = build_templates(feats, start, calib_why)
                    if built is None:
                        self._failed_feats = feats
                        self._why = "calibration check failed: " + "; ".join(calib_why)
                    else:
                        self.tpl, self.thr = built
                        try:
                            np.savez(TEMPLATE_FILE, tpl=self.tpl, thr=self.thr, ver=FEATURE_VERSION)
                        except Exception:
                            pass
                        self._grid_feats = None
                        self.calibrated_game = True
                        self.set_orientation(start)
                        self.locked = True
                        self.board = chess.Board()
                        self.last_placement = None
                        self.candidate = None
                        self.stable = 0
                        self.arrow = None
                        self.set_status("Auto-calibration done. You are playing "
                                        + ("White" if start else "Black"))
            else:
                self.calibrated_game = False

            if self.tpl is None:
                self.wait_ticks += 1
                home, stray = start_diagnosis(feats)
                if start is not None:
                    text = "Starting position seen, but " + (self._why or "calibration failed")
                elif start_why:
                    text = "Starting position seen, but " + start_why[0]
                else:
                    text = (f"Waiting for the starting position... "
                            f"(pieces seen: {home}/32, stray marks: {stray})")
                if self.wait_ticks == WAIT_DUMP_TICKS:
                    self.dump_debug(feats, "waiting for the starting position: " + text)
                if self._dump_done:
                    text += " - debug files saved next to the app"
                self.set_status(text, notice=True)
                return

            if feats is self._grid_feats:
                grid = self._grid
            else:
                why, alts = [], []
                grid = read_grid(feats, self.tpl, self.thr, self.prior_map(), why, alts)
                self._grid_feats, self._grid, self._alts = feats, grid, alts
                self._why = why[0] if why else ""
            if grid is None:
                self.bad_reads += 1
                if self.bad_reads == BAD_READS_LIMIT:
                    self.dump_debug(feats, "can't read the board: " + self._why)
                    self.set_status(BAD_TEXT + " (debug files saved next to the app)", notice=True)
                return
            self.bad_reads = 0
            if self.play_notice.startswith(BAD_TEXT):
                self.set_status("")

            if not self.locked:
                self.set_orientation(guess_orientation(grid))

            placement = placement_for(grid, self.white_bottom)
            if placement is None:
                return
            if placement != self.candidate:               # wait until the position settles
                self.candidate, self.stable = placement, 1
                return
            self.stable = min(self.stable + 1, 1000)
            if self.stable >= REBUILD_FRAMES:
                self._done_feats = feats                  # no need to analyse this frame again
            if placement == self.last_placement:
                return

            if not self.track_with_alts(grid, placement, self._alts):
                if self.stable < REBUILD_FRAMES:          # "from scratch" - only after several frames
                    return
                if not self.rebuild(placement):
                    alt = placement_for(grid, not self.white_bottom) if not self.locked else None
                    if alt is not None and (self.track_move(alt) or self.rebuild(alt)):
                        self.set_orientation(not self.white_bottom)
                        placement = alt
                    else:
                        return
            self.last_placement = placement
            self.locked = True
            self.arrow = None
            if not self.board.is_game_over() and self.board.turn == self.my_color:
                self.worker.submit(self.board.fen())
            self.update()
        except Exception as e:
            self.set_status(f"Scan error: {e}", notice=True)

    def on_engine_result(self, fen, uci):
        if fen == self.board.fen():            # ignore stale results
            self.arrow = uci
            print("Best move:", uci)
            self.update()

    # ---- mouse (setup mode only) ----
    def mousePressEvent(self, e):
        if not self.edit_mode:
            return
        if e.button() == Qt.RightButton:
            self.close()
        elif e.button() == Qt.LeftButton:
            if e.x() >= self.width() - 30 and e.y() >= self.height() - 30:
                self.resizing = True
            else:
                self.dragging = True
                self.drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self.edit_mode:
            if self.resizing:
                side = max(200, e.x(), e.y())
                self.resize(side, side)
            elif self.dragging:
                self.move(e.globalPos() - self.drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        if self.dragging or self.resizing:
            self.save_geometry()
        self.dragging = self.resizing = False

    # ---- drawing ----
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height())
        sq = side / 8.0

        if self.edit_mode:
            p.setPen(QPen(blue(220), 2))
            p.setBrush(blue(25))
            p.drawRect(0, 0, self.width() - 1, self.height() - 1)
            p.setPen(QPen(blue(90), 1))
            for i in range(1, 8):
                p.drawLine(int(i * sq), 0, int(i * sq), int(side))
                p.drawLine(0, int(i * sq), int(side), int(i * sq))
            p.setPen(Qt.NoPen)
            p.setBrush(blue(230))
            p.drawRect(self.width() - 18, self.height() - 18, 16, 16)

        err_color = QColor(255, 120, 120)
        if self.flash_text:
            self.draw_label(p, [self.flash_text], err_color if self.flash_error else None)
        elif self.edit_mode:
            self.draw_label(p, ["F8 - start/setup | F2 find board | right-click - exit"])
        elif self.play_notice:
            is_err = any(w in self.play_notice.lower() for w in ERROR_WORDS)
            self.draw_label(p, [self.play_notice], err_color if is_err else None)

        if self.arrow:
            self.draw_arrow(p, self.arrow, sq)

    def draw_label(self, p, lines, text_color=None):
        """Text centred on the board over an almost opaque dark plate."""
        side = min(self.width(), self.height())
        size = max(9, int(side / 50))
        font = QFont()
        font.setBold(True)
        while True:                      # shrink the font if the text doesn't fit
            font.setPixelSize(size)
            fm = QFontMetrics(font)
            tw = max(fm.horizontalAdvance(t) for t in lines)
            if tw <= self.width() - 30 or size <= 7:
                break
            size -= 1
        pad = 12
        lh = fm.height()
        box_w = tw + pad * 2
        box_h = lh * len(lines) + pad * 2
        x = (self.width() - box_w) / 2
        y = (self.height() - box_h) / 2
        p.setFont(font)
        p.setPen(QPen(blue(255), 1))
        p.setBrush(QColor(12, 22, 36, 235))          # almost opaque dark background
        p.drawRoundedRect(QRectF(x, y, box_w, box_h), 8, 8)
        p.setPen(text_color or QColor(255, 255, 255))
        for i, t in enumerate(lines):
            p.drawText(QRectF(x, y + pad + i * lh, box_w, lh), Qt.AlignHCenter | Qt.AlignVCenter, t)

    def draw_arrow(self, p, uci, sq):
        mv = chess.Move.from_uci(uci)
        wb = self.white_bottom
        r1, c1 = square_to_rc(mv.from_square, wb)
        r2, c2 = square_to_rc(mv.to_square, wb)
        # square highlights
        p.setPen(Qt.NoPen)
        p.setBrush(blue(80))
        p.drawRect(int(c1 * sq), int(r1 * sq), int(sq), int(sq))
        p.drawRect(int(c2 * sq), int(r2 * sq), int(sq), int(sq))
        # arrow
        a = QPointF((c1 + .5) * sq, (r1 + .5) * sq)
        b = QPointF((c2 + .5) * sq, (r2 + .5) * sq)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        head = sq * 0.4
        base = QPointF(b.x() - ux * head, b.y() - uy * head)
        p.setPen(QPen(blue(230), max(4, sq * 0.12), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(a, base)
        px, py = -uy * sq * 0.2, ux * sq * 0.2
        p.setPen(Qt.NoPen)
        p.setBrush(blue(240))
        p.drawPolygon(QPolygonF([b, QPointF(base.x() + px, base.y() + py),
                                 QPointF(base.x() - px, base.y() - py)]))

    def showEvent(self, e):
        super().showEvent(e)
        self.apply_win_flags()

    def closeEvent(self, e):
        self.save_geometry()
        self.worker.stop()
        self.worker.wait(3000)
        e.accept()
        QApplication.quit()


def ensure_admin():
    """Request administrator rights (UAC prompt) and restart the script as admin."""
    if not IS_WIN:
        return
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
    except Exception:
        return
    if getattr(sys, "frozen", False):                      # built exe
        args = sys.argv[1:]
    else:                                                  # plain .py
        args = [os.path.abspath(sys.argv[0])] + sys.argv[1:]
    params = " ".join(f'"{a}"' for a in args)
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, params, os.getcwd(), 1)
    if ret > 32:
        sys.exit(0)                                        # admin copy started, close this one
    print("Administrator rights were not granted - continuing without them")


if __name__ == "__main__":
    ensure_admin()
    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling)   # Qt coordinates = real pixels
    app = QApplication(sys.argv)
    w = ChessOverlay()
    w.show()
    sys.exit(app.exec_())