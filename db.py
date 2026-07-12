"""The database: one SQLite row per image.

pipeline: caption + tags + 3 vectors ──► [db] ──► gallery.sqlite

Each row stores the image path, its caption, its meta tags, and THREE vectors:

  image_emb  (512) - what the image LOOKS like        (vision encoder)
  text_emb   (512) - what its caption/tags MEAN        (text encoder)
  fused_emb (1024) - [image_emb ; text_emb] / sqrt(2)  (the concatenation)

Vectors are stored as raw float32 bytes (BLOB) — the same trick real vector
stores use. No extensions, no servers: plain SQLite from the standard library.
"""
import json
import sqlite3

import numpy as np

DB_PATH = "gallery.sqlite"

SCHEMA = """CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE,
    caption   TEXT,
    tags      TEXT,   -- JSON list of meta tags
    image_emb BLOB,   -- 512 x float32
    text_emb  BLOB,   -- 512 x float32
    fused_emb BLOB    -- 1024 x float32
)"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def to_blob(vec) -> bytes:
    # "<f4" = little-endian float32, spelled out so a gallery.sqlite written
    # on one machine decodes to the same floats on any other
    return np.asarray(vec, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")  # read-only view over the bytes


def add_image(con, path, caption, tags, image_emb, text_emb, fused_emb):
    con.execute(
        "INSERT OR REPLACE INTO images"
        " (path, caption, tags, image_emb, text_emb, fused_emb)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (path, caption, json.dumps(tags),
         to_blob(image_emb), to_blob(text_emb), to_blob(fused_emb)),
    )
    con.commit()


def all_images(con):
    rows = con.execute(
        "SELECT path, caption, tags, image_emb, text_emb, fused_emb FROM images"
    ).fetchall()
    return [
        {
            "path": r[0],
            "caption": r[1],
            "tags": json.loads(r[2]),
            "image_emb": from_blob(r[3]),
            "text_emb": from_blob(r[4]),
            "fused_emb": from_blob(r[5]),
        }
        for r in rows
    ]
