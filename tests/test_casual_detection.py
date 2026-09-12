"""
Unit tests for casual-message detection in src/rag_chain.py.

Run from the project root (with venv active):
    python -m unittest tests.test_casual_detection -v

These tests verify that the lightweight heuristic classifier correctly
distinguishes casual conversational messages (greetings, thanks, small
talk, identity questions) from real document-seeking questions.

The most important test class is TestCompoundMessagesAreNotCasual: it
verifies that a message which merely starts with a casual word but
contains real content ("Thanks, and what does chapter 3 say?") is NOT
misclassified as casual. If it were, the real question inside it would
never reach the retriever and the user would get an unhelpful generic
reply instead of a grounded answer.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import _is_casual_message


class TestGreetingsAreCasual(unittest.TestCase):

    def test_greetings(self):
        cases = ["hi", "Hi!", "hello", "Hello there", "hey",
                  "good morning", "Good morning!", "good evening", "good night",
                  "what's up", "whats up"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestWellbeingChecksAreCasual(unittest.TestCase):

    def test_wellbeing(self):
        cases = ["How are you?", "how r u", "how's it going", "How are you doing?"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestGratitudeAndFeedbackAreCasual(unittest.TestCase):

    def test_gratitude(self):
        cases = ["thanks", "Thanks!", "thank you", "thx", "ty",
                  "That was helpful", "that's great", "Nice", "great", "awesome",
                  "perfect", "good job", "well done"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestIdentityAndCapabilityQuestionsAreCasual(unittest.TestCase):

    def test_identity(self):
        cases = ["What are you?", "who are you", "What can you do?",
                  "what do you do", "Tell me about yourself"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestFarewellsAreCasual(unittest.TestCase):

    def test_farewells(self):
        cases = ["bye", "goodbye", "see you", "take care", "good night"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestAcknowledgmentsAreCasual(unittest.TestCase):

    def test_acknowledgments(self):
        cases = ["ok", "okay", "alright", "got it", "sure"]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertTrue(_is_casual_message(msg), f"Expected casual: {msg!r}")


class TestDocumentQuestionsAreNotCasual(unittest.TestCase):
    """The core correctness guarantee: real questions must reach the retriever."""

    def test_real_questions(self):
        cases = [
            "What is deep learning?",
            "What are the questions in chapter 1?",
            "Summarize chapter 2 for me",
            "What does the document say about recursion?",
            "Explain CNN architecture",
            "What are the questions in CO-2?",
            "How many planets are there in our solar system?",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertFalse(_is_casual_message(msg), f"Expected NOT casual: {msg!r}")


class TestCompoundMessagesAreNotCasual(unittest.TestCase):
    """
    Critical edge case: a message that STARTS with a casual word but
    contains a real question must NOT be classified as casual, or the
    real question would never reach the retriever.
    """

    def test_compound_messages(self):
        cases = [
            "Thanks, and also what does chapter 3 say about recursion?",
            "Hello, can you summarize the document for me?",
            "Hi! What are the questions in unit 1?",
            "Ok, now explain the second point in more detail.",
        ]
        for msg in cases:
            with self.subTest(msg=msg):
                self.assertFalse(_is_casual_message(msg), f"Expected NOT casual: {msg!r}")


class TestEdgeCases(unittest.TestCase):

    def test_empty_and_whitespace(self):
        for msg in ["", "   ", "\n", "\t"]:
            with self.subTest(msg=msg):
                self.assertFalse(_is_casual_message(msg))

    def test_case_insensitivity(self):
        self.assertTrue(_is_casual_message("HELLO"))
        self.assertTrue(_is_casual_message("ThAnKs"))

    def test_punctuation_tolerance(self):
        self.assertTrue(_is_casual_message("hi!!!"))
        self.assertTrue(_is_casual_message("thanks."))
        self.assertTrue(_is_casual_message("how are you???"))


if __name__ == "__main__":
    unittest.main()
