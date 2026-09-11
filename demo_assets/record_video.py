"""Render the AetherCheck replay page to an MP4.

Drives the demo page's own clock frame by frame in headless Chromium and pipes
each screenshot into ffmpeg. Because the page replays captured traces, every
number on screen is real detector output - this is a recording of the demo, not
an animation of it.

    python record_video.py <page.html> <out.mp4>
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
from playwright.sync_api import sync_playwright

FPS = 24
SPEED = 2.0                      # play the call back at 2x
STEP_MS = 1000.0 / FPS * SPEED
WIDTH = 1280

CAPTION_CSS = """
#rec-cap{
  display:flex;align-items:baseline;gap:14px;
  padding:14px 18px;margin:0 0 14px;border-radius:10px;
  background:#141923;border:1px solid #28313F;
}
#rec-cap .t{font-size:19px;font-weight:700;letter-spacing:-.01em;color:#E4E9F2}
#rec-cap .s{font-family:"IBM Plex Mono",monospace;font-size:12px;color:#98A2B6}
#rec-cap .flag{
  margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;padding:5px 10px;border-radius:5px;
  background:rgba(63,185,80,.14);color:#3FB950;border:1px solid rgba(63,185,80,.3);
}
#rec-cap .flag[data-b="elevated"]{background:rgba(214,160,40,.14);color:#D6A028;border-color:rgba(214,160,40,.35)}
#rec-cap .flag[data-b="critical"]{background:rgba(248,81,73,.16);color:#F85149;border-color:rgba(248,81,73,.45)}

/* The HUD is taller than the stage once every badge is showing, and the stage
   clips its own overflow - give it room so no badge is cut off on camera. */
.stage{min-height:530px !important}
.play{display:none !important}
"""

SETUP_JS = """
() => {
  const css = document.createElement('style');
  css.textContent = window.__CAPCSS__;
  document.head.appendChild(css);
  const cap = document.createElement('div');
  cap.id = 'rec-cap';
  cap.innerHTML = '<span class="t"></span><span class="s"></span><span class="flag"></span>';
  const tabs = document.querySelector('.tabs');
  tabs.parentNode.insertBefore(cap, tabs);
  tabs.style.display = 'none';           // the caption says which call we are on
  document.querySelector('.masthead').style.display = 'none';
  document.querySelector('.standfirst').style.display = 'none';
  document.querySelector('.verdicts').style.display = 'none';
  document.querySelector('.note').style.display = 'none';
  document.querySelector('.play').style.display = 'none';
  window.__cap = cap;
}
"""

SCENES = [
    {"key": "attack", "title": "AI voice agent — “digital arrest” scam",
     "sub": "2.0s turn gaps · Silero VAD + faster-whisper", "hold": 2.0},
    {"key": "human", "title": "Real human caller — same detector, same settings",
     "sub": "250ms turn gaps · must stay quiet", "hold": 2.5},
]


def record(page_url: str, out_path: str) -> None:
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    frames = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 900},
                                device_scale_factor=1)
        page.goto(page_url)
        page.wait_for_timeout(1800)            # let webfonts settle
        page.add_init_script("window.__CAPCSS__ = `%s`" % CAPTION_CSS)
        page.evaluate("window.__CAPCSS__ = `%s`" % CAPTION_CSS)
        page.evaluate(SETUP_JS)

        # One clip box for the whole video so nothing jumps between scenes.
        box = page.evaluate("""() => {
            const a = document.querySelector('#rec-cap').getBoundingClientRect();
            const b = document.querySelector('.timeline').getBoundingClientRect();
            const w = document.querySelector('.wrap').getBoundingClientRect();
            return {x: Math.round(w.x), y: Math.round(a.y + window.scrollY),
                    width: Math.round(w.width),
                    height: Math.round(b.bottom + window.scrollY - a.y - window.scrollY)};
        }""")
        box["height"] += 24          # the timeline's x-axis labels sit below its box
        box["width"] -= box["width"] % 2
        box["height"] -= box["height"] % 2
        print(f"[*] clip {box['width']}x{box['height']}")

        for scene in SCENES:
            page.evaluate(
                """(k) => {
                    const btn = document.querySelector('.tab[data-scenario="' + k + '"]');
                    btn.click();
                    window.__cap.querySelector('.t').textContent = '';
                }""", scene["key"])
            page.evaluate(
                """(s) => {
                    window.__cap.querySelector('.t').textContent = s.title;
                    window.__cap.querySelector('.s').textContent = s.sub;
                }""", scene)

            total = page.evaluate("dur()")
            ms = 0.0
            while ms <= total:
                page.evaluate("""(ms) => {
                    t = ms; render();
                    const f = frameAt(ms);
                    const flag = window.__cap.querySelector('.flag');
                    flag.dataset.b = f.b;
                    flag.textContent = f.b === 'critical' ? 'Scam likely \\u00b7 alert fired'
                                     : f.b === 'elevated' ? 'Elevated' : 'Clear';
                }""", ms)
                writer.append_data(imageio.imread(page.screenshot(clip=box)))
                frames += 1
                ms += STEP_MS

            # Hold on the final verdict so it is readable.
            last = imageio.imread(page.screenshot(clip=box))
            for _ in range(int(scene["hold"] * FPS)):
                writer.append_data(last)
                frames += 1
            print(f"[+] {scene['key']}: {total/1000:.0f}s of call captured")

        browser.close()
    writer.close()
    size = Path(out_path).stat().st_size / 1e6
    print(f"[=] {out_path}: {frames} frames, {frames/FPS:.1f}s, {size:.1f} MB")


if __name__ == "__main__":
    record(sys.argv[1], sys.argv[2])
