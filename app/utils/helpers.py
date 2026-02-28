CASUAL_PATTERNS = [
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "sup", "what's up", "yo", "hiya",
    "bye", "goodbye", "see you", "later", "take care", "catch you", "peace",
    "thanks", "thank you", "thx", "ty", "much appreciated", "cheers",
    "great", "awesome", "nice", "cool", "perfect", "wow", "ok", "okay", "amazing",
    "fantastic", "good job", "well done", "bravo",
    "how are you", "how's it going", "what's new", "long time no see", 
    "what's up", "all good", "no worries", "sounds good"
]

def is_casual_message(text: str) -> bool:
    """
    Returns True ONLY for clear greetings, farewells, and thanks.
    Does NOT classify short questions as casual.
    """
    clean = text.strip().lower().rstrip("!.#(),?؟")
    for pattern in CASUAL_PATTERNS:
        if clean == pattern or clean.startswith(pattern + " ") or clean.startswith(pattern + "،"):
            return True
    return False
