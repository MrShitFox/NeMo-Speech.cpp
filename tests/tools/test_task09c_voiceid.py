#!/usr/bin/env python3
"""Deterministic unit checks for Task 09C pooling and population transforms."""

from __future__ import annotations

import unittest

import numpy as np

from task09c_voiceid import (
    apply_population_transform,
    embed_pool,
    fit_diagonal_standardizer,
    fit_regularized_whitener,
    frame_statistics,
    progressive_prefix,
)
from voiceid_registry_research import pack_f32, unpack_f32


class Task09CPoolingTests(unittest.TestCase):
    def test_frame_pooling_statistics(self) -> None:
        frames = np.array([[1.0, 1.0]] * 9 + [[100.0, 100.0]], dtype=np.float32)
        stats = frame_statistics(frames)
        np.testing.assert_allclose(stats["mean"], [10.9, 10.9])
        np.testing.assert_allclose(stats["median"], [1.0, 1.0])
        np.testing.assert_allclose(stats["trimmed_mean"], [1.0, 1.0])
        self.assertAlmostEqual(float(np.linalg.norm(stats["frame_l2_mean"])), 1.0, places=6)
        self.assertEqual(embed_pool(stats, "mean").shape, (2,))
        self.assertEqual(embed_pool(stats, "mean_std").shape, (4,))

    def test_population_transforms_use_stable_training_statistics(self) -> None:
        training = np.array([[1.0, 2.0, -1.0], [3.0, 1.0, 2.0], [2.0, 5.0, 0.0],
                             [4.0, 3.0, 1.0]], dtype=np.float64)
        query = np.array([[2.5, 2.0, 1.0], [-3.0, 4.0, 2.0]], dtype=np.float32)
        diagonal_a = fit_diagonal_standardizer(training)
        diagonal_b = fit_diagonal_standardizer(training.copy())
        for key in ("center", "scale"):
            np.testing.assert_array_equal(diagonal_a[key], diagonal_b[key])
        transformed = apply_population_transform(query, diagonal_a)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1e-6)

        white_a = fit_regularized_whitener(training)
        white_b = fit_regularized_whitener(training.copy())
        np.testing.assert_array_equal(white_a["projection"], white_b["projection"])
        white_query = apply_population_transform(query, white_a)
        self.assertTrue(np.isfinite(white_query).all())
        np.testing.assert_allclose(np.linalg.norm(white_query, axis=1), 1.0, atol=1e-6)

    def test_progressive_prefix_skips_silence_and_reports_overshoot(self) -> None:
        frames = np.array([[1.0, 0.0], [100.0, 100.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        weights = np.array([0.4, 0.0, 0.4, 0.4], dtype=np.float64)
        mean, duration, count = progressive_prefix(frames, weights, 0.7)
        np.testing.assert_allclose(mean, [0.5, 0.5])
        self.assertAlmostEqual(duration, 0.8)
        self.assertEqual(count, 2)

    def test_compact_registry_float32_serialization_round_trip(self) -> None:
        prototype = np.array([0.125, -0.5, 1.0, 0.0], dtype=np.float32)
        encoded = pack_f32(prototype)
        decoded = unpack_f32(encoded, len(prototype))
        np.testing.assert_array_equal(decoded, prototype)
        self.assertEqual(len(encoded), 24)
        with self.assertRaises(ValueError):
            unpack_f32(encoded, len(prototype) + 1)


if __name__ == "__main__":
    unittest.main()
