from __future__ import annotations

import numpy as np

from quantum_crypto import BB84Protocol, E91Protocol, QuantumOTP


def run_bb84() -> None:
    print("\n" + "-" * 60)
    print("  1. BB84 — Alice and Bob over a clean channel vs. with Eve")
    print("-" * 60)
    for eve in (False, True):
        key, stats = BB84Protocol(seed=1).run_protocol(n_qubits=2000, eve_present=eve)
        label = "Eve intercepting" if eve else "Clean channel   "
        verdict = "SECURE" if stats.protocol_secure else "ABORTED"
        print(f"  {label} | sifted={stats.n_sifted_bits:4d} | QBER={stats.qber:6.2%} "
              f"| key={stats.final_key_length:4d} bits | {verdict}")
    print(f"\n  Intercept-resend pushes the QBER to ~25%, above the "
          f"{BB84Protocol.QBER_THRESHOLD:.0%} abort threshold.")


def run_e91() -> None:
    print("\n" + "-" * 60)
    print("  2. E91 — entanglement-based QKD with a Bell test")
    print("-" * 60)
    key, stats = E91Protocol(seed=3).run_protocol(n_pairs=4000)
    mismatch = np.mean(np.array(stats.alice_key) != np.array(stats.bob_key))
    print(f"  CHSH value |S| = {abs(stats.chsh_value):.3f}  "
          f"(classical bound 2, quantum bound 2√2 = {2 * np.sqrt(2):.3f})")
    print(f"  Bell inequality violated: {stats.bell_violated}")
    print(f"  Raw key: {stats.n_key_bits} bits, Alice/Bob mismatch: {mismatch:.2%}")


def run_otp() -> None:
    print("\n" + "-" * 60)
    print("  3. Quantum one-time pad — encrypt a message with a QKD key")
    print("-" * 60)
    message = "Quantum keys can't be copied 🔐 — no-cloning theorem!"
    for protocol in ("bb84", "e91"):
        result = QuantumOTP(seed=11).secure_communicate(message, protocol=protocol)
        cipher = "".join(map(str, result.ciphertext_bits[:48]))
        print(f"  [{protocol.upper()}] key bits used: {result.key_length}")
        print(f"         ciphertext: {cipher}...")
        print(f"         decrypted : {result.decrypted_text}")
        print(f"         success   : {result.success}")


def main() -> None:
    print("=" * 60)
    print("  Quantum Cryptography — Secure Channel Demo")
    print("=" * 60)
    run_bb84()
    run_e91()
    run_otp()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
