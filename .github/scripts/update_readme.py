#!/usr/bin/env python3
import os
import random
from datetime import datetime, timezone

# Lists of interesting quantum quotes and facts to select from
QUANTUM_QUOTES = [
    {
        "quote": "If quantum mechanics hasn't profoundly shocked you, you haven't understood it yet.",
        "author": "Niels Bohr"
    },
    {
        "quote": "I think I can safely say that nobody understands quantum mechanics.",
        "author": "Richard Feynman"
    },
    {
        "quote": "God does not play dice with the universe.",
        "author": "Albert Einstein"
    },
    {
        "quote": "Quantum physics thus reveals a basic oneness of the universe.",
        "author": "Erwin Schrödinger"
    },
    {
        "quote": "The bit is the quantum of information.",
        "author": "Anton Zeilinger"
    },
    {
        "quote": "Those who are not shocked when they first come across quantum theory cannot possibly have understood it.",
        "author": "Niels Bohr"
    },
    {
        "quote": "We must be clear that when it comes to atoms, language can be used only as in poetry. The poet, too, is not nearly so concerned with describing facts as with creating images and establishing mental connections.",
        "author": "Niels Bohr"
    }
]

QUANTUM_FACTS = [
    "Quantum superposition allows a qubit to be in states |0⟩ and |1⟩ simultaneously until it is measured.",
    "Quantum entanglement is what Albert Einstein famously referred to as 'spooky action at a distance' (spukhafte Fernwirkung).",
    "Shor's algorithm can factor integers in polynomial time, posing a fundamental challenge to modern RSA encryption.",
    "Grover's algorithm provides a quadratic speedup for searching an unsorted database of size N, taking O(√N) time.",
    "The No-Cloning Theorem states that it is impossible to create an identical copy of an arbitrary unknown quantum state.",
    "Quantum tunneling is a phenomenon where a particle penetrates a potential energy barrier that it classically couldn't cross.",
    "Quantum decoherence is the loss of quantum coherence, where a system's state behaves classically due to interaction with the environment.",
    "In a Bloch sphere representation, the poles represent the pure states |0⟩ and |1⟩, while the equator represents equal superpositions."
]

def generate_random_state():
    """Generates a random single-qubit quantum state representation."""
    import math
    # Generate random angles for Bloch sphere
    # theta in [0, pi], phi in [0, 2*pi]
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    
    # Calculate amplitudes
    alpha = math.cos(theta / 2.0)
    beta_real = math.sin(theta / 2.0) * math.cos(phi)
    beta_imag = math.sin(theta / 2.0) * math.sin(phi)
    
    # Format representation
    # |ψ⟩ = alpha|0⟩ + (beta_r + i*beta_i)|1⟩
    sign = "+" if beta_imag >= 0 else "-"
    beta_str = f"({beta_real:.3f} {sign} {abs(beta_imag):.3f}i)"
    
    return f"|ψ⟩ = {alpha:.3f}|0⟩ + {beta_str}|1⟩"

def main():
    # Resolve README.md path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.normpath(os.path.join(script_dir, "../../README.md"))
    
    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}")
        return
        
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    start_tag = "<!-- START_SECTION:update -->"
    end_tag = "<!-- END_SECTION:update -->"
    
    if start_tag not in content or end_tag not in content:
        print("Error: Placeholder tags not found in README.md")
        return
        
    # Pick a random quote and fact
    quote_obj = random.choice(QUANTUM_QUOTES)
    fact = random.choice(QUANTUM_FACTS)
    qubit_state = generate_random_state()
    
    # Get current timestamp
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Format the updated block beautifully
    update_content = (
        f"\n"
        f"> **Daily Quote:** *\"{quote_obj['quote']}\"* — **{quote_obj['author']}**\n"
        f">\n"
        f"> **Quantum Fact of the Day:** {fact}\n"
        f">\n"
        f"> **Today's Qubit State:** `{qubit_state}`\n"
        f">\n"
        f"> 🕒 *Last Quantum State Update: {timestamp_str}*\n"
    )
    
    # Replace content between tags
    start_idx = content.find(start_tag) + len(start_tag)
    end_idx = content.find(end_tag)
    
    new_content = content[:start_idx] + update_content + content[end_idx:]
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Successfully updated README.md with timestamp: {timestamp_str}")

if __name__ == "__main__":
    main()
