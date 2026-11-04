"""
TikTok slider-captcha handling for the BD brand scraper.

It's the SAME "drag the puzzle piece into place" secsdk slider the creator
scraper solves (same #captcha-verify-image bg, .captcha_verify_img_slide piece,
.secsdk-captcha-drag-icon handle). So the PRIMARY solver here is the creator
scraper's proven one: correlate the puzzle piece's EDGE pattern against the
background's edge map (normalized cross-correlation) to locate the notch — robust
on busy natural-photo backgrounds where single-strongest-edge picking drifts.

Order of attempts per puzzle:
  1. edge-template match (creator's _gap_x_by_template) on the piece vs background
  2. edge-energy strongest column (creator's _find_captcha_gap) as a cheap fallback
  3. pristine-fetch + desaturated-notch detection as a last resort (kept from the
     BD-specific solver — helps if a screenshot ever composites badly)
A low-confidence read on all of them reshuffles to a fresh puzzle instead of
burning a bad drag. Then a short manual-solve window in the headed browser.

Set CAPTCHA_DUMP=1 to save each attempt's bg + estimate under logs/captcha_samples/.
"""
import math
import os
import time

from playwright.async_api import Page

CAPTCHA_SELS = [
    '[class*="captcha"]', '[id*="captcha"]', '[class*="puzzle"]', '[class*="secsdk"]',
]
BG_SELS = ['#captcha-verify-image', 'img#captcha-verify-image',
           '[class*="captcha"] img', '[class*="verify"] img']
PIECE_SELS = ['.captcha_verify_img_slide', '[class*="img_slide"]']
HANDLE_SELS = ['.secsdk-captcha-drag-icon', '.captcha_verify_slide--button',
               '[class*="drag-icon"]', '[class*="captcha"] [class*="drag"]']

# below this template correlation the notch read is unreliable — fall back to the
# other detectors, and if they're weak too, reshuffle rather than drag blindly.
_TEMPLATE_MIN = 0.28
# Silhouette-match confidence gate: peak-margin below this means the piece SHAPE
# wasn't clearly found on the bg (busy scenery) — reshuffle instead of dragging.
# Measured on real captures: correct reads scored 0.24–0.34, wrong ones ~0.02.
_SIL_CONF_GATE = 0.15
_SEARCH_LO, _SEARCH_HI = 0.06, 0.98
_W_DESAT, _W_RIM = 0.6, 0.4
_CONF_GATE = 0.35
_DUMP = os.environ.get("CAPTCHA_DUMP") == "1"


def _log(msg):
    print(f"[captcha] {msg}", flush=True)


async def captcha_present(page: Page) -> bool:
    for sel in CAPTCHA_SELS:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                return True
        except Exception:
            pass
    # A real captcha ALWAYS renders the DOM elements matched above. The old
    # body-text fallback matched generic words ("Geser" = Indonesian "slide",
    # "Verify"…) that ALSO appear in normal pool/product content once scrolling
    # loads it — a permanent FALSE POSITIVE that made the sweep "detect" a captcha
    # with no slider on every filter step (→ handle-not-found loops, skipped
    # filters, dropdowns that open-but-never-pick). Only trust an unmistakable
    # full captcha phrase here, never a lone generic word.
    try:
        body = await page.inner_text("body")
        if any(k in body for k in ("Drag the puzzle piece", "Verify you are human")):
            # corroborate with a captcha/verify element so a stray phrase in product
            # copy can't trip it either
            for sel in ('[class*="secsdk"]', '#captcha-verify-image', '[class*="captcha_verify"]'):
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    return True
    except Exception:
        pass
    return False


async def _first_visible(page, selectors):
    for sel in selectors:
        try:
            h = await page.query_selector(sel)
            if h and await h.is_visible():
                return h
        except Exception:
            pass
    return None


def _gap_x_by_template(bg_png, piece_png, dy_frac, exclude_frac):
    """Creator scraper's matcher: correlate the piece's EDGE pattern against the
    background's edge map (the notch retains the piece's outline). Returns
    (gap_center_frac, score) or (None, 0.0). Score is a normalized correlation in
    [-1, 1]; a clean match sits ~0.4–0.8."""
    try:
        from PIL import Image
        import numpy as np
        import io
        B = np.asarray(Image.open(io.BytesIO(bg_png)).convert("L"), dtype=np.float64)
        P = np.asarray(Image.open(io.BytesIO(piece_png)).convert("L"), dtype=np.float64)
        Bh, Bw = B.shape
        Ph, Pw = P.shape
        if Pw < 8 or Pw >= Bw or Ph < 8 or Ph > Bh:
            return None, 0.0

        def edges(a):
            e = np.zeros_like(a)
            e[:, :-1] += np.abs(np.diff(a, axis=1))
            e[:-1, :] += np.abs(np.diff(a, axis=0))
            return e

        Be, Pe = edges(B), edges(P)
        dy = int(max(0, min(Bh - Ph, round(dy_frac * Bh))))
        band = Be[dy:dy + Ph, :]
        if band.shape[0] < Ph:
            return None, 0.0
        Pn = Pe - Pe.mean()
        pden = float(np.sqrt((Pn * Pn).sum())) or 1.0
        excl = int(exclude_frac * Bw)
        best_x, best = None, -1e18
        for x in range(0, Bw - Pw):
            if abs(x - excl) < Pw:               # skip the piece's own current position
                continue
            w = band[:, x:x + Pw]
            wn = w - w.mean()
            wden = float(np.sqrt((wn * wn).sum())) or 1.0
            score = float((wn * Pn).sum()) / (wden * pden)   # normalized correlation
            if score > best:
                best, best_x = score, x
        if best_x is None:
            return None, 0.0
        return (best_x + Pw / 2.0) / Bw, float(best)
    except Exception:
        return None, 0.0


def _locate_gap_silhouette(bg_png, piece_png, bg_box, piece_box):
    """PRIMARY solver: match the pristine PIECE's white-border silhouette against the
    pristine background's notch outline (normalized edge cross-correlation) at the
    piece's y-band. Robust to scenery/clouds/sun because it matches the puzzle SHAPE,
    not brightness/saturation. Returns (gap_center_frac, confidence) or (None, 0.0);
    confidence is the peak-to-runner-up margin — gate with _SIL_CONF_GATE."""
    try:
        from PIL import Image
        import numpy as np
        import io
        bg = np.asarray(Image.open(io.BytesIO(bg_png)).convert("L"), dtype=np.float64)
        Bh, Bw = bg.shape
        parr = np.asarray(Image.open(io.BytesIO(piece_png)).convert("RGBA"), dtype=np.float64)
        alpha = parr[:, :, 3]
        lum = parr[:, :, :3].mean(axis=2)
        Pw = max(8, min(int(round(piece_box["width"] / bg_box["width"] * Bw)), Bw - 2))
        Ph = max(8, int(round(alpha.shape[0] * Pw / alpha.shape[1])))
        # template = the piece's bright WHITE RIM (drawn identically around the bg notch)
        wb = (((lum > 200) & (alpha > 128)).astype(np.float64)) * 255.0
        tmpl = np.asarray(Image.fromarray(wb).resize((Pw, Ph)), dtype=np.float64)

        def edges(a):
            e = np.zeros_like(a)
            e[:, :-1] += np.abs(np.diff(a, axis=1))
            e[:-1, :] += np.abs(np.diff(a, axis=0))
            return e

        Be, Pe = edges(bg), edges(tmpl)
        dy = int(round((piece_box["y"] - bg_box["y"]) / bg_box["height"] * Bh))
        dy = max(0, min(Bh - Ph, dy))
        band = Be[dy:dy + Ph, :]
        if band.shape[0] < Ph:
            return None, 0.0
        Pn = Pe - Pe.mean()
        pden = float(np.sqrt((Pn * Pn).sum())) or 1.0
        excl = int(round((piece_box["x"] - bg_box["x"]) / bg_box["width"] * Bw))
        allscores, best_x, best = [], None, -1e18
        for x in range(0, Bw - Pw):
            if abs(x - excl) < Pw * 0.6:                 # skip the piece's own position
                continue
            w = band[:, x:x + Pw]
            wn = w - w.mean()
            wden = float(np.sqrt((wn * wn).sum())) or 1.0
            s = float((wn * Pn).sum()) / (wden * pden)
            allscores.append((x, s))
            if s > best:
                best, best_x = s, x
        if best_x is None:
            return None, 0.0
        far = [s for (x, s) in allscores if abs(x - best_x) >= Pw]
        conf = float(best - (max(far) if far else 0.0))
        return (best_x + Pw / 2.0) / Bw, max(0.0, conf)
    except Exception:
        return None, 0.0


def _find_captcha_gap(png_bytes, piece_left_frac=0.12):
    """Creator scraper's cheap fallback: strongest vertical-edge column."""
    try:
        from PIL import Image
        import numpy as np
        import io
        a = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"), dtype=np.int32)
        h, w = a.shape
        if w < 40:
            return None
        col = np.abs(a[:, 1:] - a[:, :-1]).sum(axis=0).astype(float)
        lo, hi = int(w * piece_left_frac), int(w * 0.95)
        if hi - lo < 5:
            return None
        sm = np.convolve(col[lo:hi], np.ones(7) / 7.0, mode="same")
        return (lo + int(np.argmax(sm))) / float(w)
    except Exception:
        return None


async def _fetch_pristine(page, el):
    """Fetch an element's `src` image bytes (no screenshot compositing)."""
    try:
        src = await el.get_attribute("src")
        if not src:
            return None, None
        resp = await page.request.get(src)
        return await resp.body(), src
    except Exception:
        return None, None


def _locate_gap_pristine(bg_png, bg_box, piece_box):
    """Last-resort BD-specific fallback: notch by low colour-saturation + white-rim
    on the pristine background. Returns (frac, confidence) or (None, 0.0)."""
    try:
        from PIL import Image
        import numpy as np
        import io
        rgb = np.asarray(Image.open(io.BytesIO(bg_png)).convert("RGB"), dtype=np.float64)
        Bh, Bw, _ = rgb.shape
        lum = rgb.mean(axis=2)
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        Pw = int(round(piece_box["width"] / bg_box["width"] * Bw))
        Pw = max(8, min(Pw, Bw - 2))
        dy = int(round((piece_box["y"] - bg_box["y"]) / bg_box["height"] * Bh))
        dy = max(0, min(Bh - Pw, dy))
        excl = int(round((piece_box["x"] - bg_box["x"]) / bg_box["width"] * Bw))
        band_sat = sat[dy:dy + Pw, :]
        band_lum = lum[dy:dy + Pw, :]
        x0 = max(0, int(_SEARCH_LO * Bw))
        x1 = min(Bw - Pw, int(_SEARCH_HI * Bw))
        xs = [x for x in range(x0, x1) if abs(x - excl) >= Pw * 0.6]
        if len(xs) < 3:
            return None, 0.0
        desat = np.array([-band_sat[:, x:x + Pw].mean() for x in xs])
        rim = np.array([(band_lum[:, x:x + Pw] > 235).mean() for x in xs])

        def _norm(v):
            lo, hi = float(v.min()), float(v.max())
            return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

        combined = _norm(desat) * _W_DESAT + _norm(rim) * _W_RIM
        bi = int(np.argmax(combined))
        best_x = xs[bi]
        far = [combined[j] for j, x in enumerate(xs) if abs(x - best_x) >= Pw]
        conf = float(combined[bi] - (max(far) if far else 0.0))
        return (best_x + Pw / 2.0) / Bw, max(0.0, min(1.0, conf))
    except Exception:
        return None, 0.0


async def _human_drag(page: Page, handle, dx: float):
    """Drag `handle` right by dx px with an eased, slightly jittered, human path."""
    box = await handle.bounding_box()
    if not box:
        return
    sx, sy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await page.mouse.move(sx, sy)
    await page.mouse.down()
    steps = 30
    for i in range(1, steps + 1):
        t = i / steps
        ease = 0.5 - 0.5 * math.cos(math.pi * t)         # smooth accel → decel
        await page.mouse.move(sx + dx * ease, sy + math.sin(t * math.pi) * 2.5)
        await page.wait_for_timeout(10 + int(10 * (1 - abs(0.5 - t) * 2)))
    await page.mouse.move(sx + dx + 2, sy)               # tiny overshoot + correct
    await page.wait_for_timeout(60)
    await page.mouse.move(sx + dx, sy)
    await page.wait_for_timeout(120)
    await page.mouse.up()


def _dump_sample(bg_png, gap_frac, conf, dx, src):
    try:
        d = os.path.join(os.path.dirname(__file__), "logs", "captcha_samples")
        os.makedirs(d, exist_ok=True)
        stamp = str(int(time.time() * 1000))
        if bg_png:
            with open(os.path.join(d, f"{stamp}_bg.png"), "wb") as f:
                f.write(bg_png)
        with open(os.path.join(d, f"{stamp}.txt"), "w") as f:
            f.write(f"gap_frac={gap_frac} conf={conf} dx={dx} src={src}\n")
    except Exception:
        pass


async def _auto_solve_slider(page: Page) -> bool:
    """Best-effort auto-solve of the 'drag the puzzle piece' captcha, using the
    creator scraper's edge-template matcher first, then cheaper fallbacks. Returns
    True if cleared."""
    # A captcha that pops mid-sweep (after a filter click) renders its slider a beat
    # late — the bg/piece <img>s aren't in the DOM the instant captcha_present() goes
    # true. Retry the element hunt with short settle-waits before giving up, else
    # every quick attempt bails at this guard and the whole captcha "misses".
    handle = bg = piece = None
    for _ in range(6):
        handle = handle or await _first_visible(page, HANDLE_SELS)
        bg = bg or await _first_visible(page, BG_SELS)
        piece = piece or await _first_visible(page, PIECE_SELS)
        if handle and bg and piece:
            break
        await page.wait_for_timeout(700)
    if not handle:
        _log("drag handle not found yet (slider still rendering / variant) — retry next round")
        return False
    if not bg:
        _log("bg image not found yet — retry next round")
        return False
    bg_box = await bg.bounding_box()
    piece_box = await piece.bounding_box() if piece else None
    if not bg_box or not bg_box["width"] or not bg_box["height"]:
        return False

    gap_frac, method, score, dumped = None, "none", 0.0, None
    bg_png, _bsrc = await _fetch_pristine(page, bg)
    piece_png = None
    if piece:
        piece_png, _psrc = await _fetch_pristine(page, piece)
    # 1) PRIMARY: silhouette-match the pristine piece's shape against the pristine bg
    if bg_png and piece_png and piece_box:
        gf, conf = _locate_gap_silhouette(bg_png, piece_png, bg_box, piece_box)
        if gf is not None and conf >= _SIL_CONF_GATE:
            gap_frac, method, score, dumped = gf, "silhouette", conf, bg_png
    # 2) desaturated-notch on the pristine bg
    if gap_frac is None and bg_png and piece_box:
        gf, conf = _locate_gap_pristine(bg_png, bg_box, piece_box)
        if gf is not None and conf >= _CONF_GATE:
            gap_frac, method, score, dumped = gf, "pristine-desat", conf, bg_png
    # 3) crude strongest-edge column on a screenshot — last resort only
    if gap_frac is None:
        bg_shot = await bg.screenshot()
        gf = _find_captcha_gap(bg_shot)
        if gf is not None:
            gap_frac, method, dumped = gf, "edge-energy", bg_shot
    if gap_frac is None:
        _log("no confident notch read — reshuffling")
        return False

    gap_x = bg_box["x"] + gap_frac * bg_box["width"]
    piece_cx = (piece_box["x"] + piece_box["width"] / 2) if piece_box else (bg_box["x"] + bg_box["width"] * 0.06)
    dx = max(8.0, gap_x - piece_cx)
    _log(f"auto-solve: gap~{gap_frac:.2f} drag~{int(dx)}px (via {method}, score={score:.2f})")
    if _DUMP:
        _dump_sample(dumped, gap_frac, score, int(dx), method)
    await _human_drag(page, handle, dx)
    await page.wait_for_timeout(1600)
    return not await captcha_present(page)


async def handle_captcha(page: Page, auto_attempts: int = 6, manual_wait_s: int = 120) -> bool:
    """Detect the captcha; try to auto-solve the slider several times (reshuffling
    a fresh puzzle between misses), then fall back to a short manual-solve window.
    Returns True when cleared; no-op (True) when none is present."""
    if not await captcha_present(page):
        return True

    _log("detected — attempting auto-solve")
    await page.wait_for_timeout(1200)          # let the slider widget finish rendering
    for attempt in range(auto_attempts):
        try:
            if await _auto_solve_slider(page):
                _log(f"auto-solved ✓ (attempt {attempt + 1})")
                await page.wait_for_timeout(800)
                return True
        except Exception as e:
            _log(f"auto-solve error: {str(e)[:80]}")
        await page.wait_for_timeout(1000)
        if not await captcha_present(page):
            return True
        # reshuffle to a fresh background (a new image often reads cleaner)
        try:
            btn = await page.query_selector('[class*="refresh"], [class*="Refresh"], .secsdk_captcha_refresh')
            if btn and await btn.is_visible():
                await btn.click(timeout=2000)
                await page.wait_for_timeout(1200)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  CAPTCHA — auto-solve missed. Solve it in the window if you're there;")
    print(f"  otherwise it auto-skips this filter in ~{manual_wait_s}s and continues.")
    print("=" * 60 + "\n", flush=True)
    rounds = max(1, manual_wait_s // 2)
    for _ in range(rounds):
        await page.wait_for_timeout(2000)
        if not await captcha_present(page):
            _log("solved — continuing")
            await page.wait_for_timeout(800)
            return True
    _log("not solved in time — skipping this filter")
    return False
