# captcha

TikTok secsdk slider-captcha ("drag the puzzle piece") solver for Playwright (async).

```python
from captcha import handle_captcha
ok = await handle_captcha(page)   # True when cleared (or no captcha present)
```

Solve order: piece-silhouette edge match → desaturated-notch on pristine bg → strongest-edge column fallback; low-confidence reads reshuffle instead of dragging. Set `CAPTCHA_DUMP=1` to save samples under `logs/captcha_samples/`.

Requires: `pip install -r requirements.txt`
