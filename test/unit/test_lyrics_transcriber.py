# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Lyrics axis scoring, text sanitization and ASR gating in lyrics.lyrics_transcriber.

Covers the pinned 27-label axis column order, per-axis softmax scoring of an
embedding, the cleanup applied to raw lyric text before embedding, and the
transcript quality gate that reads the ASR backend's reported confidence.

Main Features:
* axis_columns is a pure, duplicate-free 27-tuple in canonical axis/label order
* _score_axes returns one softmax chunk per axis that sums to 1 and whose argmax
  points at the targeted label
* _sanitize_lyrics_text strips control/zero-width chars, HTML, and LRC timestamps
  and truncates to a word cap; _softmax sums to one and is temperature-monotonic
* _asr_confidence reads avg_logprob as unknown (None) when a backend omits it or
  reports it non-numerically, and _asr_should_drop then skips the confidence
  gates instead of dropping every transcript
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope='module')
def lt():
    return importlib.import_module('lyrics.lyrics_transcriber')


class TestAxisColumns:
    EXPECTED_TOTAL_LABELS = 27

    EXPECTED_ORDER = (
        ('AXIS_1_SETTING', 'URBAN'),
        ('AXIS_1_SETTING', 'WILDERNESS'),
        ('AXIS_1_SETTING', 'INTERIOR'),
        ('AXIS_1_SETTING', 'TRANSIT'),
        ('AXIS_1_SETTING', 'EXTRATERRESTRIAL'),
        ('AXIS_1_SETTING', 'SURREAL_ABSTRACT'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'SOLITARY'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'ROMANTIC'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'KINSHIP'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'COLLECTIVE'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'ADVERSARIAL'),
        ('AXIS_2_SOCIAL_DYNAMIC', 'DIVINE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'RADIANT'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'MELANCHOLIC'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'VOLATILE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'VULNERABLE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'SERENE'),
        ('AXIS_3_EMOTIONAL_VALENCE', 'NUMB'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'RETROSPECTIVE'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'CHRONICLE'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'EXISTENTIAL'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'STORYTELLING'),
        ('AXIS_4_NARRATIVE_TEMPORALITY', 'DIRECT_PLEA'),
        ('AXIS_5_THEMATIC_WEIGHT', 'TRIVIAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'MORTAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'POLITICAL'),
        ('AXIS_5_THEMATIC_WEIGHT', 'SENSORIAL'),
    )

    def test_total_length_matches_sum_of_labels(self, lt):
        expected = sum(len(meta['labels']) for meta in lt.MUSIC_ANALYSIS_AXES.values())
        assert len(lt.axis_columns()) == expected

    def test_total_length_is_exactly_27(self, lt):
        assert len(lt.axis_columns()) == self.EXPECTED_TOTAL_LABELS

    def test_canonical_order_is_pinned(self, lt):
        assert tuple(lt.axis_columns()) == self.EXPECTED_ORDER

    def test_order_is_axes_then_labels_in_definition_order(self, lt):
        cols = lt.axis_columns()
        idx = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            for label in meta['labels'].keys():
                assert cols[idx] == (axis_name, label), (
                    f'Position {idx}: expected {(axis_name, label)}, got {cols[idx]}'
                )
                idx += 1
        assert idx == len(cols)


class TestScoreAxes:
    @staticmethod
    def _fake_axis_state(lt):
        rng = np.random.default_rng(42)
        label_map = {}
        axis_embeddings = {}
        emb_dim = 8
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            labels = [(name, name.lower()) for name in meta['labels'].keys()]
            label_map[axis_name] = labels
            mat = rng.standard_normal((len(labels), emb_dim)).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            axis_embeddings[axis_name] = mat
        return label_map, axis_embeddings, emb_dim

    def test_vector_length_matches_axis_columns(self, lt):
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings', return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        assert vec.dtype == np.float32
        assert vec.shape == (len(lt.axis_columns()),)
        assert vec.shape == (27,)

    def test_each_axis_chunk_is_softmax_summing_to_one(self, lt):
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings', return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        offset = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            n = len(meta['labels'])
            chunk = vec[offset : offset + n]
            assert np.all(chunk >= 0.0), f'{axis_name}: negative softmax entry'
            assert np.all(chunk <= 1.0), f'{axis_name}: softmax entry > 1'
            assert chunk.sum() == pytest.approx(1.0, abs=1e-5), (
                f'{axis_name}: chunk sum {chunk.sum()} != 1.0'
            )
            offset += n
        assert offset == vec.shape[0]

    def test_chunk_layout_aligns_with_axis_columns(self, lt):
        label_map, axis_embeddings, emb_dim = self._fake_axis_state(lt)
        embedding = np.ones(emb_dim, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)

        with patch.object(lt, '_get_axis_embeddings', return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(embedding)

        cols = lt.axis_columns()
        by_axis: dict[str, list[int]] = {}
        for i, (axis_name, _label) in enumerate(cols):
            by_axis.setdefault(axis_name, []).append(i)
        prev_end = 0
        for axis_name in lt.MUSIC_ANALYSIS_AXES.keys():
            indices = by_axis[axis_name]
            assert indices == list(range(prev_end, prev_end + len(indices)))
            assert vec[indices].sum() == pytest.approx(1.0, abs=1e-5)
            prev_end += len(indices)

    def test_argmax_per_axis_points_to_targeted_label(self, lt):
        rng = np.random.default_rng(7)
        emb_dim = 8
        query = rng.standard_normal(emb_dim).astype(np.float32)
        query /= np.linalg.norm(query)

        winners = {
            'AXIS_1_SETTING': 'TRANSIT',
            'AXIS_2_SOCIAL_DYNAMIC': 'DIVINE',
            'AXIS_3_EMOTIONAL_VALENCE': 'RADIANT',
            'AXIS_4_NARRATIVE_TEMPORALITY': 'EXISTENTIAL',
            'AXIS_5_THEMATIC_WEIGHT': 'POLITICAL',
        }

        label_map = {}
        axis_embeddings = {}
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            labels = [(name, name.lower()) for name in meta['labels'].keys()]
            label_map[axis_name] = labels
            mat = rng.standard_normal((len(labels), emb_dim)).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            target_idx = [name for name, _ in labels].index(winners[axis_name])
            mat[target_idx] = query
            axis_embeddings[axis_name] = mat

        with patch.object(lt, '_get_axis_embeddings', return_value=(label_map, axis_embeddings)):
            vec = lt._score_axes(query)

        cols = lt.axis_columns()
        offset = 0
        for axis_name, meta in lt.MUSIC_ANALYSIS_AXES.items():
            n = len(meta['labels'])
            chunk = vec[offset : offset + n]
            argmax_global = offset + int(np.argmax(chunk))
            actual_axis, actual_label = cols[argmax_global]
            assert actual_axis == axis_name, f'Argmax for {axis_name} landed in {actual_axis}'
            assert actual_label == winners[axis_name], (
                f'{axis_name}: expected winning label {winners[axis_name]!r}, '
                f'got {actual_label!r} (vector ordering mismatch)'
            )
            offset += n


class TestSanitizeLyricsText:
    def test_empty_input_returns_empty(self, lt):
        assert lt._sanitize_lyrics_text('') == ''
        assert lt._sanitize_lyrics_text(None) == ''

    def test_strips_control_chars_and_zero_width(self, lt):
        text = 'hello\u200bworld\ufefftest\x00\x07end'
        out = lt._sanitize_lyrics_text(text)
        assert '\u200b' not in out
        assert '\ufeff' not in out
        assert '\x00' not in out
        assert '\x07' not in out
        assert 'hello' in out and 'world' in out and 'end' in out

    def test_strips_html_tags(self, lt):
        text = 'line one <b>bold</b> line\n<script>alert(1)</script>after'
        out = lt._sanitize_lyrics_text(text)
        assert '<b>' not in out and '</b>' not in out
        assert '<script>' not in out
        assert 'alert(1)' not in out
        assert 'bold' in out
        assert 'line one' in out and 'after' in out

    def test_truncates_to_max_words(self, lt):
        text = ' '.join(['word'] * 500)
        out = lt._sanitize_lyrics_text(text, max_words=100)
        assert len(out.split()) == 100

    def test_collapses_blank_line_runs(self, lt):
        text = 'a\n\n\n\nb\n\n\n\n\nc'
        out = lt._sanitize_lyrics_text(text)
        lines = out.split('\n')
        for i in range(len(lines) - 1):
            if lines[i] == '' and lines[i + 1] == '':
                pytest.fail(f'Multiple consecutive blank lines: {lines!r}')


class TestStripLrcTimestamps:
    def test_strips_leading_timestamps(self, lt):
        lrc = '[00:01.23]first line\n[00:05.67]second line'
        out = lt._strip_lrc_timestamps(lrc)
        assert out == 'first line\nsecond line'

    def test_drops_empty_lines_after_stripping(self, lt):
        lrc = '[00:01.00]\n[00:02.00]only content\n[00:03.00]'
        out = lt._strip_lrc_timestamps(lrc)
        assert out == 'only content'


class TestAsrConfidence:
    """A plugin ASR backend may not report avg_logprob at all - see _asr_backend."""

    def test_reads_the_reported_value(self, lt):
        assert lt._asr_confidence({'avg_logprob': -0.25}) == pytest.approx(-0.25)

    def test_missing_key_is_unknown(self, lt):
        assert lt._asr_confidence({'text': 'hi', 'language': 'en'}) is None

    def test_explicit_none_is_unknown(self, lt):
        assert lt._asr_confidence({'avg_logprob': None}) is None

    def test_non_numeric_is_unknown_and_warns(self, lt, caplog):
        assert lt._asr_confidence({'avg_logprob': 'very good'}) is None
        assert 'non-numeric avg_logprob' in caplog.text

    def test_nan_is_unknown_and_warns_instead_of_disabling_the_gate(self, lt, caplog):
        assert lt._asr_confidence({'avg_logprob': float('nan')}) is None
        assert 'NaN avg_logprob' in caplog.text

    def test_overflowing_value_is_unknown_not_fatal(self, lt):
        assert lt._asr_confidence({'avg_logprob': 10 ** 400}) is None

    def test_neg_inf_is_the_builtin_worst_confidence_sentinel_and_stays_real(self, lt):
        assert lt._asr_confidence({'avg_logprob': float('-inf')}) == float('-inf')

    def test_pos_inf_stays_real_but_warns_as_impossible(self, lt, caplog):
        assert lt._asr_confidence({'avg_logprob': float('inf')}) == float('inf')
        assert 'cannot drop this transcript' in caplog.text

    def test_zero_the_documented_forbidden_placeholder_is_kept_but_warned(self, lt, caplog):
        assert lt._asr_confidence({'avg_logprob': 0.0}) == 0.0
        assert 'cannot drop this transcript' in caplog.text

    def test_good_negative_confidence_does_not_warn(self, lt, caplog):
        assert lt._asr_confidence({'avg_logprob': -0.3}) == pytest.approx(-0.3)
        assert 'cannot drop this transcript' not in caplog.text

    def test_postprocess_keeps_a_transcript_without_confidence(self, lt):
        text = (
            'I walked alone into the pouring rain and thought about the days '
            'we spent together, the summer light, the empty street, the sound '
            'of your name, the songs we sang out loud until the morning came '
            'and left us nothing but the echo of a promise we could not keep.'
        )
        raw_text, raw_len, lang, logprob, detected = lt._postprocess_asr(
            {'text': text, 'language': 'en'}
        )
        assert raw_text == text
        assert raw_len == len(text)
        assert lang == 'en'
        assert detected == 'en'
        assert logprob is None


class TestAsrShouldDrop:
    GOOD_TEXT = 'a real transcript'

    def _drop(self, lt, lang, logprob, text=None):
        text = self.GOOD_TEXT if text is None else text
        return lt._asr_should_drop(text, len(text), lang, logprob)

    def test_drops_low_confidence(self, lt):
        assert self._drop(lt, 'en', lt.ASR_MIN_AVG_LOGPROB - 0.1) is True

    def test_keeps_good_confidence(self, lt):
        assert self._drop(lt, 'en', lt.ASR_MIN_AVG_LOGPROB + 0.1) is False

    def test_neg_inf_confidence_still_drops_it_is_not_an_unknown(self, lt):
        assert self._drop(lt, 'en', float('-inf')) is True

    def test_unknown_confidence_skips_the_gate(self, lt):
        assert self._drop(lt, 'en', None) is False

    def test_unknown_confidence_skips_the_non_english_gate(self, lt):
        assert self._drop(lt, 'de', None) is False

    def test_drops_low_confidence_non_english(self, lt):
        logprob = lt.ASR_NON_ENGLISH_MIN_LOGPROB - 0.1
        assert self._drop(lt, 'de', logprob) is True

    def test_null_language_still_drops_without_confidence(self, lt):
        assert self._drop(lt, 'nospeech', None) is True

    def test_empty_text_is_never_dropped_here(self, lt):
        assert self._drop(lt, 'en', float('-inf'), text='') is False


class TestSoftmax:
    def test_sums_to_one(self, lt):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = lt._softmax(v, temperature=0.5)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_monotonic_under_positive_temperature(self, lt):
        v = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = lt._softmax(v, temperature=0.1)
        assert out[0] < out[1] < out[2]

    def test_empty_input_returns_empty(self, lt):
        v = np.array([], dtype=np.float32)
        out = lt._softmax(v, temperature=0.1)
        assert out.size == 0


class TestNarrativeChunking:
    def test_chunk_text_splits_and_keeps_everything(self, lt):
        text = 'x' * 4000
        chunks = lt._chunk_text(text, 1800)
        assert [len(c) for c in chunks] == [1800, 1800, 400]
        assert ''.join(chunks) == text

    def test_chunk_text_splits_text_without_spaces(self, lt):
        assert len(lt._chunk_text('\u4f60\u597d' * 1500, 1800)) == 2

    def test_embed_chunked_weights_chunks_by_length(self, lt):
        vecs = iter([np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)])
        with patch.object(lt, 'NARRATIVE_CHUNK_CHARS', 30), patch.object(
            lt, '_embed_text', side_effect=lambda *a, **k: next(vecs)
        ):
            out = lt._embed_chunked('a' * 40, None, None)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
        assert out[0] > out[1] > 0

    def test_embed_chunked_none_when_nothing_embeds(self, lt):
        with patch.object(lt, '_embed_text', return_value=None):
            assert lt._embed_chunked('abc', None, None) is None


class TestWindowLanguages:
    def test_unsure_windows_take_nearest_confident_language(self):
        from lyrics.whisper_onnx import fill_window_languages

        got = fill_window_languages([('nn', 0.3), ('en', 0.9), ('nn', 0.4), ('it', 0.8), ('en', 0.2)], 0.7)
        assert got == ['en', 'en', 'en', 'it', 'it']

    def test_none_when_no_window_is_confident(self):
        from lyrics.whisper_onnx import fill_window_languages

        assert fill_window_languages([('nn', 0.3), ('de', 0.5)], 0.7) is None


class TestTitleBlend:
    def test_title_only_when_no_transcript_embedding(self, lt):
        t = np.array([0, 1], dtype=np.float32)
        assert lt._blend_title(None, t) is t

    def test_blend_is_unit_and_leans_to_transcript(self, lt):
        e = np.array([1, 0], dtype=np.float32)
        t = np.array([0, 1], dtype=np.float32)
        with patch.object(lt, 'NARRATIVE_TITLE_WEIGHT', 0.3):
            out = lt._blend_title(e, t)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
        assert out[0] > out[1] > 0


class TestFinalizeEmbeddingTitle:
    def test_no_speech_uses_title_vector_not_sentinel(self, lt):
        title_vec = np.array([0, 1], dtype=np.float32)
        with patch.object(lt, 'NARRATIVE_MODE', True), patch.object(
            lt, 'load_topic_embedding_model', return_value=(None, None)
        ), patch.object(lt, '_embed_text', return_value=title_vec), patch.object(
            lt, '_score_axes', return_value=np.zeros(3, dtype=np.float32)
        ):
            emb, axes = lt._finalize_embedding('', '[F4M] Some Title')
        assert emb is title_vec
        assert axes.shape == (3,)

    def test_music_mode_ignores_title(self, lt):
        with patch.object(lt, 'NARRATIVE_MODE', False), patch.object(
            lt, '_make_instrumental_sentinel', return_value=(np.ones(2, dtype=np.float32), np.zeros(1, dtype=np.float32))
        ):
            emb, _ = lt._finalize_embedding('', 'Some Title')
        assert emb.tolist() == [1.0, 1.0]


class TestPerWindowLanguageDecoding:
    def test_each_window_is_decoded_with_its_own_language(self):
        from lyrics import whisper_onnx as w

        pipe = object.__new__(w._OnnxWhisperPipeline)
        pipe.mel_filters = None
        tokens = []

        def process(enc, idx, n, samples, token, max_new):
            tokens.append(token)
            return 'text', -0.3

        audio = np.concatenate(
            [np.full(w.WHISPER_CHUNK_SAMPLES, 1.0, np.float32), np.full(w.WHISPER_CHUNK_SAMPLES, 2.0, np.float32)]
        )
        with patch.object(w, 'NARRATIVE_MODE', True), patch.object(
            w, '_log_mel_spectrogram', lambda x, f: x
        ), patch.object(pipe, '_encode', lambda m: m), patch.object(
            pipe, '_detect_language', lambda enc: ('en', 0.9) if enc[0] == 1.0 else ('it', 0.9)
        ), patch.object(pipe, '_resolve_lang_token_id', lambda code: code), patch.object(
            pipe, '_process_chunk', side_effect=process
        ):
            out = pipe.transcribe(audio)
        assert tokens == ['en', 'it']
        assert out['text'] == 'text text'


class TestFindSimilarByItem:
    def _mgr(self):
        from tasks import lyrics_manager

        return lyrics_manager

    def test_item_with_only_the_placeholder_vector_has_no_neighbours(self):
        m = self._mgr()
        placeholder = np.zeros(4, dtype=np.float32)
        placeholder[0] = 1.0
        with patch.dict(m._LYRICS_INDEX_CACHE, {'loaded': True, 'index': object()}), patch.object(
            m, '_item_vector', return_value=None
        ), patch.object(m, '_search_by_vector') as search:
            assert m.find_similar_by_item('a', 5) == []
        search.assert_not_called()

    def test_seed_is_excluded_and_limit_is_passed(self):
        m = self._mgr()
        vec = np.array([0.5, 0.5], dtype=np.float32)
        with patch.dict(m._LYRICS_INDEX_CACHE, {'loaded': True, 'index': object()}), patch.object(
            m, '_item_vector', return_value=vec
        ), patch.object(m, '_search_by_vector', return_value=[{'item_id': 'b'}]) as search:
            assert m.find_similar_by_item('a', 7) == [{'item_id': 'b'}]
        assert search.call_args.args[1] == 7
        assert search.call_args.kwargs['seed']['item_id'] == 'a'

    def test_empty_when_index_not_loaded(self):
        m = self._mgr()
        with patch.dict(m._LYRICS_INDEX_CACHE, {'loaded': False, 'index': None}):
            assert m.find_similar_by_item('a', 5) == []


class TestNarrativeFloor:
    def test_saved_music_values_cannot_cut_narrative_mode_short(self, monkeypatch):
        import config

        monkeypatch.setattr(config, 'LYRICS_NARRATIVE_MODE', True)
        monkeypatch.setattr(config, 'AUDIO_LOAD_TIMEOUT', 600)
        monkeypatch.setattr(config, 'LYRICS_MUSICNN_SKIP', True)
        config._apply_narrative_floor()
        assert config.AUDIO_LOAD_TIMEOUT == 7200
        assert config.LYRICS_MUSICNN_SKIP is False

    def test_a_larger_timeout_is_kept_and_music_mode_is_untouched(self, monkeypatch):
        import config

        monkeypatch.setattr(config, 'LYRICS_NARRATIVE_MODE', True)
        monkeypatch.setattr(config, 'AUDIO_LOAD_TIMEOUT', 10000)
        config._apply_narrative_floor()
        assert config.AUDIO_LOAD_TIMEOUT == 10000
        monkeypatch.setattr(config, 'LYRICS_NARRATIVE_MODE', False)
        monkeypatch.setattr(config, 'AUDIO_LOAD_TIMEOUT', 600)
        monkeypatch.setattr(config, 'LYRICS_MUSICNN_SKIP', True)
        config._apply_narrative_floor()
        assert config.AUDIO_LOAD_TIMEOUT == 600 and config.LYRICS_MUSICNN_SKIP is True
