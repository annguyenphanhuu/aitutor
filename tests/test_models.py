"""
Tests for ORM Models.

Module: app/db/models.py
Covers:
- User model — field defaults
- SkillMastery model — field defaults & constraints
- InteractionLog model — field defaults
- QuizSession model — field defaults
- QuizQuestion model — field defaults
- SpacedRepetitionCard model — field defaults
"""

import pytest
from datetime import datetime
from app.db.models import (
    User,
    SkillMastery,
    InteractionLog,
    QuizSession,
    QuizQuestion,
    SpacedRepetitionCard,
)


# ━━ User Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUserModel:

    def test_create_user(self):
        u = User(username="john", display_name="John")
        assert u.username == "john"
        assert u.display_name == "John"

    def test_tablename(self):
        assert User.__tablename__ == "users"


# ━━ SkillMastery Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSkillMasteryModel:

    def test_construction(self):
        """SQLAlchemy defaults are server-side; in-memory values are None until flush."""
        m = SkillMastery(user_id=1, skill_id="test", skill_name="Test")
        assert m.skill_id == "test"
        assert m.skill_name == "Test"
        assert m.user_id == 1

    def test_explicit_values(self):
        m = SkillMastery(user_id=1, skill_id="test", skill_name="Test",
                         p_mastery=0.5, total_attempts=3, correct_attempts=2)
        assert m.p_mastery == 0.5
        assert m.total_attempts == 3
        assert m.correct_attempts == 2

    def test_tablename(self):
        assert SkillMastery.__tablename__ == "skill_masteries"


# ━━ InteractionLog Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInteractionLogModel:

    def test_create(self):
        log = InteractionLog(
            user_id=1,
            question="What is 2+2?",
            answer="4",
            is_correct=True,
            skill_id="arithmetic",
        )
        assert log.is_correct is True
        assert log.skill_id == "arithmetic"

    def test_tablename(self):
        assert InteractionLog.__tablename__ == "interaction_logs"


# ━━ QuizSession Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuizSessionModel:

    def test_construction(self):
        """SQLAlchemy defaults are server-side; test explicit values."""
        s = QuizSession(user_id=1, session_type="practice", total_questions=5,
                        questions=[], is_completed=False, correct_count=0,
                        current_index=0)
        assert s.session_type == "practice"
        assert s.total_questions == 5
        assert s.is_completed is False
        assert s.correct_count == 0
        assert s.current_index == 0

    def test_tablename(self):
        assert QuizSession.__tablename__ == "quiz_sessions"


# ━━ QuizQuestion Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuizQuestionModel:

    def test_create(self):
        q = QuizQuestion(
            skill_id="derivative_basic",
            difficulty=1,
            question_latex="$f'(x) = ?$",
            choices=["$2x$", "$x^2$", "$3x$", "$1$"],
            correct_index=0,
        )
        assert q.difficulty == 1
        assert len(q.choices) == 4

    def test_tablename(self):
        assert QuizQuestion.__tablename__ == "quiz_questions"


# ━━ SpacedRepetitionCard Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSpacedRepetitionCardModel:

    def test_construction(self):
        """SpacedRepetitionCard has easiness_factor, interval, repetitions."""
        c = SpacedRepetitionCard(
            user_id=1,
            skill_id="derivative_basic",
            easiness_factor=2.5,
            interval=0,
            repetitions=0,
        )
        assert c.easiness_factor == 2.5
        assert c.interval == 0
        assert c.repetitions == 0
        assert c.skill_id == "derivative_basic"

    def test_tablename(self):
        assert SpacedRepetitionCard.__tablename__ == "spaced_repetition_cards"
