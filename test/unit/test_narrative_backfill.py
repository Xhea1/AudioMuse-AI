# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>


"""The narrative backfill script is safe by default and reversible.

Main Features:
* A dry run reports but never writes
* --apply backs rows up before deleting them, and refuses when narrative mode is off
* A placeholder-only vector is recognised
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location('narrative_backfill', REPO / 'scripts' / 'narrative_backfill.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.sql = []
        self.full = []
        self.backup_exists = False
        self.read_only = None

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def fetchone(self):
        return ('lyrics_embedding_pre_narrative',) if self.backup_exists else (None,)

    def execute(self, sql, params=None):
        self.full.append(sql)
        self.sql.append(sql.split()[0] + ' ' + ' '.join(sql.split()[1:4]))

    def fetchall(self):
        return self.rows

    def commit(self):
        self.sql.append('COMMIT')

    def close(self):
        pass


def _rows():
    real = np.array([0.5, 0.5], dtype=np.float32).tobytes()
    placeholder = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    return [('a', 'T1', 'X', 1500.0, real), ('b', 'T2', 'Y', 600.0, placeholder)]


def _run(script, monkeypatch, argv, narrative=True):
    conn = FakeConn(_rows())
    monkeypatch.setattr(script, 'connect_raw', lambda read_only=False: setattr(conn, 'read_only', read_only) or conn)
    monkeypatch.setattr(script.config, 'LYRICS_NARRATIVE_MODE', narrative)
    return script.main(argv), conn


def test_placeholder_vector_is_recognised(script):
    assert script.is_placeholder(np.array([1.0, 0.0], dtype=np.float32).tobytes())
    assert not script.is_placeholder(np.array([0.5, 0.5], dtype=np.float32).tobytes())
    assert script.is_placeholder(None)


def test_dry_run_reads_only_and_writes_nothing(script, monkeypatch, capsys):
    code, conn = _run(script, monkeypatch, [])
    assert code == 0 and conn.read_only is True
    assert not any(s.startswith(('DELETE', 'INSERT', 'CREATE', 'COMMIT')) for s in conn.sql)
    assert '2 tracks selected, 1 hold only' in capsys.readouterr().out


def test_apply_backs_up_before_deleting(script, monkeypatch):
    code, conn = _run(script, monkeypatch, ['--apply'])
    steps = [s.split()[0] for s in conn.sql]
    assert code == 0 and conn.read_only is False
    assert steps.index('INSERT') < steps.index('DELETE') < steps.index('COMMIT')
    assert 'CREATE' in steps


def test_apply_refuses_when_narrative_mode_is_off(script, monkeypatch):
    code, conn = _run(script, monkeypatch, ['--apply'], narrative=False)
    assert code == 2
    assert conn.sql == []


def test_tracks_already_in_the_backup_table_are_skipped(script, monkeypatch):
    conn = FakeConn(_rows())
    conn.backup_exists = True
    monkeypatch.setattr(script, 'connect_raw', lambda read_only=False: conn)
    monkeypatch.setattr(script.config, 'LYRICS_NARRATIVE_MODE', True)
    script.main(['--limit', '5'])
    select = next(q for q in conn.full if q.lstrip().startswith('SELECT s.item_id'))
    assert 'NOT EXISTS' in select and 'lyrics_embedding_pre_narrative' in select


def test_first_run_has_no_backup_table_to_exclude(script, monkeypatch):
    code, conn = _run(script, monkeypatch, [])
    select = next(q for q in conn.full if q.lstrip().startswith('SELECT s.item_id'))
    assert 'NOT EXISTS' not in select
