from __future__ import annotations

import numpy as np
import pytest

from quantum_crypto import BB84Protocol, E91Protocol, QuantumOTP


def test_bb84_clean_channel_produces_matching_keys():
    key, stats = BB84Protocol(seed=0).run_protocol(n_qubits=2000)
    assert stats.protocol_secure
    assert stats.qber == 0.0
    assert stats.keys_match
    assert stats.final_key_length == len(key) > 256


def test_bb84_detects_intercept_resend():
    key, stats = BB84Protocol(seed=0).run_protocol(n_qubits=2000, eve_present=True)
    assert not stats.protocol_secure
    assert 0.15 < stats.qber < 0.35
    assert len(key) == 0


def test_sifting_keeps_about_half():
    _, stats = BB84Protocol(seed=1).run_protocol(n_qubits=4000)
    assert stats.n_sifted_bits / 4000 == pytest.approx(0.5, abs=0.05)


def test_privacy_amplification_is_deterministic_per_seed(rng):
    key = rng.integers(0, 2, size=300)
    a = BB84Protocol.privacy_amplification(key, 0.5, seed=9)
    b = BB84Protocol.privacy_amplification(key, 0.5, seed=9)
    c = BB84Protocol.privacy_amplification(key, 0.5, seed=10)
    assert len(a) == 150
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(np.unique(a)) <= {0, 1}


def test_e91_violates_chsh_and_keys_agree():
    _, stats = E91Protocol(seed=0).run_protocol(n_pairs=6000)
    assert stats.bell_violated
    assert abs(stats.chsh_value) == pytest.approx(2 * np.sqrt(2), abs=0.15)
    assert stats.alice_key == stats.bob_key


@pytest.mark.parametrize("protocol", ["bb84", "e91"])
@pytest.mark.parametrize("message", ["hi", "Halo dunia 🚀 — ünïcödé ok", "x" * 120])
def test_otp_roundtrip(protocol, message):
    result = QuantumOTP(seed=4).secure_communicate(message, protocol=protocol)
    assert result.success, result.error_message
    assert result.decrypted_text == message


def test_otp_rejects_unknown_protocol():
    result = QuantumOTP().secure_communicate("x", protocol="rsa")
    assert not result.success


def test_otp_encrypt_requires_long_enough_key():
    with pytest.raises(ValueError):
        QuantumOTP.encrypt(np.ones(16, dtype=int), np.ones(8, dtype=int))
