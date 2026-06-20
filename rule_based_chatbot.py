"""
Project 1: Rule-Based AI Chatbot
=================================

Goal:
    Create a simple rule-based chatbot that responds to predefined
    user inputs using if-else decision logic, running in a continuous
    loop until the user exits.

Key Requirements covered:
    1. Handle greetings and exit commands
    2. Use if-else logic for responses
    3. Run in a continuous loop

Key Skills demonstrated:
    Control flow, decision-making logic, basic AI concepts (pattern
    matching against a rule base instead of machine learning).

Author: (Your Name Here)
"""

import random
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. RULE BASE
# ---------------------------------------------------------------------------
# A "rule" here is simply: a set of trigger keywords -> a set of possible
# responses. This is the core of a rule-based system: no learning, no
# probability model, just direct pattern matching and decision logic.
# Using lists of responses (instead of a single string) lets the bot feel
# less robotic, since it picks a random reply each time a rule matches.

RULES = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"],
        "responses": [
            "Hello there! How can I help you today?",
            "Hi! What can I do for you?",
            "Hey! Good to see you.",
        ],
    },
    "how_are_you": {
        "keywords": ["how are you", "how's it going", "how are u"],
        "responses": [
            "I'm just a bunch of if-else statements, but I'm doing great! How about you?",
            "Running smoothly, thanks for asking!",
        ],
    },
    "name": {
        "keywords": ["your name", "who are you", "what are you"],
        "responses": [
            "I'm RuleBot, a simple rule-based chatbot built in Python.",
            "You can call me RuleBot!",
        ],
    },
    "time": {
        "keywords": ["time", "what time is it"],
        "responses": ["dynamic_time"],  # handled specially below
    },
    "help": {
        "keywords": ["help", "what can you do", "options", "commands"],
        "responses": [
            "I can greet you, tell you the time, answer simple questions, "
            "and chat a bit. Try saying 'hello', 'what's the time', or 'bye'!",
        ],
    },
    "thanks": {
        "keywords": ["thank you", "thanks", "appreciate it"],
        "responses": [
            "You're very welcome!",
            "No problem at all!",
            "Anytime!",
        ],
    },
    "weather": {
        "keywords": ["weather", "raining", "sunny", "temperature outside"],
        "responses": [
            "I don't have live weather data, but I hope it's nice outside!",
        ],
    },
    "joke": {
        "keywords": ["joke", "make me laugh", "funny"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the function break up with the loop? It needed space to return.",
        ],
    },
}

# Exit commands are handled separately from RULES because they need to
# break the main loop rather than just print a response.
EXIT_COMMANDS = ["bye", "exit", "quit", "goodbye", "see you", "stop"]

FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you rephrase that?",
    "Hmm, I don't have a rule for that yet. Try 'help' to see what I can do.",
    "Sorry, that's outside what I've been programmed to handle.",
]


# ---------------------------------------------------------------------------
# 2. CORE DECISION LOGIC
# ---------------------------------------------------------------------------
def get_response(user_input: str) -> str:
    """
    Core if-else decision engine of the chatbot.

    Takes the raw user input, normalizes it, and checks it against
    every rule's keyword list. Returns the first matching rule's
    response (chosen randomly from that rule's response options),
    or a fallback message if nothing matches.
    """
    text = user_input.lower().strip()

    # --- Exit check happens first so 'bye' never gets misclassified ---
    if any(cmd in text for cmd in EXIT_COMMANDS):
        return "EXIT"

    # --- Empty input edge case ---
    if text == "":
        return "Did you mean to say something? I'm listening!"

    # --- Main rule matching using if-else / elif chain ---
    if any(keyword in text for keyword in RULES["greeting"]["keywords"]):
        return random.choice(RULES["greeting"]["responses"])

    elif any(keyword in text for keyword in RULES["how_are_you"]["keywords"]):
        return random.choice(RULES["how_are_you"]["responses"])

    elif any(keyword in text for keyword in RULES["name"]["keywords"]):
        return random.choice(RULES["name"]["responses"])

    elif any(keyword in text for keyword in RULES["time"]["keywords"]):
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    elif any(keyword in text for keyword in RULES["help"]["keywords"]):
        return random.choice(RULES["help"]["responses"])

    elif any(keyword in text for keyword in RULES["thanks"]["keywords"]):
        return random.choice(RULES["thanks"]["responses"])

    elif any(keyword in text for keyword in RULES["weather"]["keywords"]):
        return random.choice(RULES["weather"]["responses"])

    elif any(keyword in text for keyword in RULES["joke"]["keywords"]):
        return random.choice(RULES["joke"]["responses"])

    else:
        # No rule matched -> fallback response
        return random.choice(FALLBACK_RESPONSES)


# ---------------------------------------------------------------------------
# 3. CONTINUOUS LOOP (the chatbot's main runtime)
# ---------------------------------------------------------------------------
def run_chatbot() -> None:
    """
    Runs the chatbot in a continuous loop:
        1. Prompt the user for input
        2. Pass input to the decision engine
        3. Print the response
        4. Repeat until an exit command is received
    """
    print("=" * 50)
    print(" RuleBot: A Simple Rule-Based AI Chatbot")
    print("=" * 50)
    print("RuleBot: Hello! Type 'help' to see what I can do, or 'bye' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nRuleBot: Goodbye! (input stream closed)")
            break

        response = get_response(user_input)

        if response == "EXIT":
            print("RuleBot: Goodbye! Have a great day. 👋")
            break

        print(f"RuleBot: {response}")


# ---------------------------------------------------------------------------
# 4. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_chatbot()
