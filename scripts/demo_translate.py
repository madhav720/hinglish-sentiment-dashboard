"""
Interactive CLI Demo: Hinglish to Hindi and English Translation
Usage:
    python scripts/demo_translate.py
    python scripts/demo_translate.py "Ye movie bilkul bakwaas hai, paisa aur time dono barbaad!"
    python scripts/demo_translate.py --interactive
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.translation.translator import translate_hinglish
from src.data.annotation_engine import score_text_sentiment
from src.utils.constants import ID2LABEL

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def predict_sentiment_quick(text: str):
    scores = score_text_sentiment(text)
    if scores["net_neg"] > scores["net_pos"]:
        label_id = 0 if scores["strong_neg_hits"] > 0 else 1
    elif scores["net_pos"] > scores["net_neg"]:
        label_id = 4 if scores["strong_pos_hits"] > 0 else 3
    else:
        label_id = 2
    return ID2LABEL[label_id]


def display_translation(text: str, idx: int = None):
    res = translate_hinglish(text)
    sentiment = predict_sentiment_quick(text)

    header = f" Example {idx} " if idx else " Custom Input "
    print(f"\n{BOLD}{CYAN}{'='*25}{header}{'='*25}{RESET}")
    print(f"{BOLD}⚡ Original Hinglish :{RESET} {YELLOW}{res['hinglish']}{RESET}")
    print(f"{BOLD}🇮🇳 Hindi (हिंदी)     :{RESET} {GREEN}{res['hindi']}{RESET}")
    print(f"{BOLD}🇬🇧 English (EN)     :{RESET} {CYAN}{res['english']}{RESET}")
    print(f"{BOLD}🏷️ Sentiment         :{RESET} {MAGENTA}{sentiment}{RESET}")
    print(f"{CYAN}{'-'*60}{RESET}")


def run_demo():
    print(f"\n{BOLD}{GREEN}{'='*60}")
    print("      HINGLISH DUAL TRANSLATION (HINDI & ENGLISH) DEMO")
    print(f"{'='*60}{RESET}")
    print("Translating Romanized Hinglish text into Devanagari Hindi and English:\n")

    curated_samples = [
        "Ye movie bilkul bakwaas hai, paisa aur time dono barbaad!",
        "Ye product bahut accha hai aur smoothly kaam karta hai",
        "Main aaj ghar ja raha hoon, apne dost se milne",
        "Boring service aur ghatiya khana, yahan kabhi mat aao",
        "Zabardast acting aur shandar kahani, superhit film hai!",
        "Is problem ka solution bahut aasan hai, koshish karo",
    ]

    for i, s in enumerate(curated_samples, 1):
        display_translation(s, idx=i)

    print(f"\n{BOLD}💡 Tip:{RESET} You can translate any custom text by running:")
    print(f"   {CYAN}python scripts/demo_translate.py \"Aapka text yahan\"{RESET}")
    print(f"Or start interactive prompt with:")
    print(f"   {CYAN}python scripts/demo_translate.py --interactive{RESET}\n")


def run_interactive():
    print(f"\n{BOLD}{GREEN}{'='*60}")
    print("   INTERACTIVE HINGLISH TRANSLATOR (Type 'quit' or 'exit' to end)")
    print(f"{'='*60}{RESET}\n")
    while True:
        try:
            user_input = input(f"{BOLD}Enter Hinglish text > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting translator. Goodbye!")
                break
            display_translation(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting translator.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["--interactive", "-i"]:
            run_interactive()
        elif arg in ["--demo", "-d"]:
            run_demo()
        elif arg in ["--text", "-t"]:
            custom_sentence = " ".join(sys.argv[2:])
            display_translation(custom_sentence)
        else:
            custom_sentence = " ".join(sys.argv[1:])
            display_translation(custom_sentence)
    else:
        run_demo()
