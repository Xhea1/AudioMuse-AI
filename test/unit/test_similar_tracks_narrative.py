# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>


"""In narrative mode /api/similar_tracks answers from the transcript index.

The Navidrome plugin's default engine calls this endpoint; the MusiCNN index it
used to search carries no content signal for speech.

Main Features:
* A found work list is returned with distance derived from similarity
* No neighbours is a 404, and the MusiCNN search is never consulted
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

pytest.importorskip('app_ivf')
from app_ivf import ivf_bp  # noqa: E402


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(ivf_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _get(client, found):
    with patch('app_ivf.LYRICS_NARRATIVE_MODE', True), patch(
        'app_server_context.resolve_input_item_id', return_value='a'
    ), patch('tasks.lyrics_manager.find_similar_by_item', return_value=found), patch(
        'app_ivf._serialize_neighbor_results', side_effect=lambda rows, n: rows
    ), patch('app_ivf.find_nearest_neighbors_by_id') as musicnn:
        response = client.get('/api/similar_tracks?item_id=a&n=3')
    return response, musicnn


def test_neighbours_come_from_the_transcript_index(client):
    response, musicnn = _get(client, [{'item_id': 'b', 'similarity': 0.8}])
    assert response.status_code == 200
    assert response.get_json() == [{'item_id': 'b', 'distance': pytest.approx(0.2)}]
    musicnn.assert_not_called()


def test_no_neighbours_is_a_404_without_falling_back_to_musicnn(client):
    response, musicnn = _get(client, [])
    assert response.status_code == 404
    musicnn.assert_not_called()
