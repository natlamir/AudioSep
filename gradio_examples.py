from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent

EXAMPLES_DIR = CURR_DIR / "examples"

EXAMPLES = [
    [EXAMPLES_DIR / "acoustic_guitar.wav", "acoustic guitar"],
    [EXAMPLES_DIR / "laughing.wav", "laughing"],
    [
        EXAMPLES_DIR / "ticktok_piano.wav",
        "A ticktock sound playing at the same rhythm with piano.",
    ],
    [EXAMPLES_DIR / "water_drops.wav", "water drops"],
    [EXAMPLES_DIR / "noisy_speech.wav", "speech"],
]
