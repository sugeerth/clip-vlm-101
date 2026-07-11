"""Download the example images (Wikimedia Commons, small 384px thumbnails).

Each entry is a well-known freely-licensed Commons file. Failures are skipped,
so a renamed file never breaks the pipeline. Credits: see README.md.
"""
import pathlib
import ssl
import time
import urllib.parse
import urllib.request

try:  # macOS system Python ships without CA certs; certifi fills the gap
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

SAMPLES = [
    ("Cat03.jpg", "cat.jpg"),
    ("Labrador Retriever portrait.jpg", "dog.jpg"),
    ("2010-kodiak-bear-1.jpg", "bear.jpg"),
    ("Ara ararauna Luc Viatour.jpg", "parrot.jpg"),
    ("Hopetoun falls.jpg", "waterfall.jpg"),
    ("Fronalpstock big.jpg", "mountains.jpg"),
    ("Tour Eiffel Wikimedia Commons.jpg", "eiffel_tower.jpg"),
    ("Palace of Westminster, London - Feb 2007.jpg", "london.jpg"),
    ("Eq it-na pizza-margherita sep2005 sml.jpg", "pizza.jpg"),
    ("Red Apple.jpg", "apple.jpg"),
    ("Sunflower sky backdrop.jpg", "sunflower.jpg"),
    ("Pluto in True Color - High-Res.jpg", "pluto.jpg"),
    ("Left side of Flying Pigeon.jpg", "bicycle.jpg"),
    ("Broadway tower edit.jpg", "castle.jpg"),
]

OUT = pathlib.Path("images")
URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{name}?width=384"


def main():
    OUT.mkdir(exist_ok=True)
    ok = 0
    for commons_name, save_as in SAMPLES:
        if (OUT / save_as).exists():
            ok += 1
            print(f"  have {save_as}")
            continue
        time.sleep(2)  # be polite: the thumbnail service rate-limits bursts
        url = URL.format(name=urllib.parse.quote(commons_name))
        req = urllib.request.Request(url, headers={"User-Agent": "clip-vlm-101/1.0"})
        try:
            data = urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT).read()
            (OUT / save_as).write_bytes(data)
            ok += 1
            print(f"  ok  {save_as}  ({len(data) // 1024} KB)")
        except Exception as e:  # skip and continue — samples are best-effort
            print(f"  SKIP {save_as}: {e}")
    print(f"{ok}/{len(SAMPLES)} images in {OUT}/")


if __name__ == "__main__":
    main()
