#!/usr/bin/env python3
# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Queue already-analysed tracks for a narrative-mode lyrics re-run.

Analysis plans each stage per track, so a track with no lyrics_embedding row
re-runs only the lyrics stage on the next analysis run; its MusiCNN and CLAP
results are kept. This script removes the old (first-4-minutes) lyrics vectors
so that happens, after copying them to a backup table.

Main Features:
* Dry run by default: lists the tracks, how many hold only the no-speech
  placeholder, the audio hours, and an estimate from --rtf (measure it on the
  target machine; 0.12 is what a fast desktop CPU gave).
* --apply backs the rows up to lyrics_embedding_pre_narrative, then deletes
  them; it refuses to run unless LYRICS_NARRATIVE_MODE is on, because the re-run
  would otherwise rebuild the same 4-minute vectors.
* --restore puts back backed-up rows that have not been recomputed yet.
* --limit, --author and --min-duration select a small test batch first.

After --apply, start a normal analysis run; the lyrics index is rebuilt by it.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import config
from database import connect_raw

BACKUP = 'lyrics_embedding_pre_narrative'


def is_placeholder(blob):
    vec = np.frombuffer(bytes(blob), dtype=np.float32) if blob is not None else np.zeros(0)
    return vec.size == 0 or not vec[1:].any()


def select_rows(cur, args):
    sql = (
        "SELECT s.item_id, s.title, s.author, COALESCE(s.duration, 0), le.embedding "
        "FROM lyrics_embedding le JOIN score s USING (item_id) "
        "WHERE COALESCE(s.duration, 0) >= %s"
    )
    params = [args.min_duration]
    if args.author:
        sql += " AND s.author ILIKE %s"
        params.append(args.author)
    sql += " ORDER BY s.item_id"
    if args.limit:
        sql += " LIMIT %s"
        params.append(args.limit)
    cur.execute(sql, params)
    return cur.fetchall()


def report(rows, rtf):
    seconds = sum(float(r[3]) for r in rows)
    placeholders = sum(1 for r in rows if is_placeholder(r[4]))
    print(f"{len(rows)} tracks selected, {placeholders} hold only the no-speech placeholder")
    print(f"audio: {seconds / 3600:.1f} h; transcription at RTF {rtf}: about {seconds * rtf / 3600:.1f} h")
    for item_id, title, author, duration, _ in rows[:10]:
        print(f"  {item_id}  {float(duration) / 60:5.1f} min  {author} - {title}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")


def apply(conn, cur, ids):
    cur.execute(f"CREATE TABLE IF NOT EXISTS {BACKUP} (LIKE lyrics_embedding INCLUDING ALL)")
    cur.execute(
        f"INSERT INTO {BACKUP} SELECT * FROM lyrics_embedding WHERE item_id = ANY(%s) "
        "ON CONFLICT (item_id) DO NOTHING",
        (ids,),
    )
    cur.execute("DELETE FROM lyrics_embedding WHERE item_id = ANY(%s)", (ids,))
    conn.commit()


def restore(conn, cur):
    cur.execute(
        f"INSERT INTO lyrics_embedding SELECT b.* FROM {BACKUP} b JOIN score s USING (item_id) "
        "ON CONFLICT (item_id) DO NOTHING"
    )
    restored = cur.rowcount
    conn.commit()
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--apply', action='store_true', help='back up and delete the selected vectors')
    parser.add_argument('--restore', action='store_true', help='put back backed-up vectors not yet recomputed')
    parser.add_argument('--limit', type=int, default=0, help='only the first N tracks')
    parser.add_argument('--author', default='', help='only this author (case-insensitive, exact)')
    parser.add_argument('--min-duration', type=float, default=0.0, help='only tracks at least this many seconds long')
    parser.add_argument('--rtf', type=float, default=0.12, help='seconds of processing per second of audio')
    args = parser.parse_args(argv)

    if args.apply and args.restore:
        parser.error('--apply and --restore cannot be combined')
    if args.apply and not config.LYRICS_NARRATIVE_MODE:
        print('Refusing: LYRICS_NARRATIVE_MODE is off, the re-run would rebuild the old 4-minute vectors.')
        return 2

    conn = connect_raw(read_only=not (args.apply or args.restore))
    try:
        with conn.cursor() as cur:
            if args.restore:
                print(f"restored {restore(conn, cur)} vectors")
                return 0
            rows = select_rows(cur, args)
            report(rows, args.rtf)
            if not args.apply:
                print('dry run: nothing changed. Add --apply to back up and delete these vectors.')
                return 0
            if rows:
                apply(conn, cur, [r[0] for r in rows])
            print(f"backed up and removed {len(rows)} vectors; start an analysis run to recompute them.")
            return 0
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
