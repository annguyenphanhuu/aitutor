"""
Tests for Knowledge Tracing Service (DB-backed).

Module: app/knowledge_tracing/service.py
Covers:
- get_or_create_mastery() — create and retrieve mastery records
- update_mastery() — BKT update through DB
- get_all_masteries() — bulk mastery retrieval
- get_mastery_profile() — full profile with levels
- identify_gaps() — prerequisite gap identification
- get_recent_results() — recent interaction history
"""

import pytest
from app.knowledge_tracing.service import (
    get_or_create_mastery,
    update_mastery,
    get_all_masteries,
    get_mastery_profile,
    identify_gaps,
    get_recent_results,
)
from app.db.models import SkillMastery, InteractionLog


# ━━ get_or_create_mastery() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetOrCreateMastery:

    @pytest.mark.asyncio
    async def test_creates_new_mastery(self, db_session):
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery is not None
        assert mastery.skill_id == "derivative_basic"
        assert mastery.p_mastery == 0.1  # BKT default p_init
        assert mastery.total_attempts == 0

    @pytest.mark.asyncio
    async def test_returns_existing_mastery(self, db_session):
        # Create first
        m1 = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        m1.p_mastery = 0.8
        await db_session.flush()

        # Get again
        m2 = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert m2.p_mastery == 0.8
        assert m1.id == m2.id

    @pytest.mark.asyncio
    async def test_different_users_different_records(self, db_session):
        m1 = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        m2 = await get_or_create_mastery(db_session, "derivative_basic", user_id=2)
        assert m1.id != m2.id

    @pytest.mark.asyncio
    async def test_skill_name_from_graph(self, db_session):
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery.skill_name == "Đạo hàm cơ bản"

    @pytest.mark.asyncio
    async def test_unknown_skill_uses_id_as_name(self, db_session):
        mastery = await get_or_create_mastery(db_session, "unknown_skill_xyz", user_id=1)
        assert mastery.skill_name == "unknown_skill_xyz"


# ━━ update_mastery() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUpdateMastery:

    @pytest.mark.asyncio
    async def test_correct_increases_mastery(self, db_session):
        initial = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        initial_p = initial.p_mastery

        new_p = await update_mastery(db_session, "derivative_basic", is_correct=True, user_id=1)
        assert new_p > initial_p

    @pytest.mark.asyncio
    async def test_increments_total_attempts(self, db_session):
        await update_mastery(db_session, "derivative_basic", is_correct=True, user_id=1)
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery.total_attempts == 1

    @pytest.mark.asyncio
    async def test_correct_increments_correct_attempts(self, db_session):
        await update_mastery(db_session, "derivative_basic", is_correct=True, user_id=1)
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery.correct_attempts == 1

    @pytest.mark.asyncio
    async def test_incorrect_does_not_increment_correct(self, db_session):
        await update_mastery(db_session, "derivative_basic", is_correct=False, user_id=1)
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery.correct_attempts == 0
        assert mastery.total_attempts == 1

    @pytest.mark.asyncio
    async def test_multiple_updates_accumulate(self, db_session):
        for _ in range(5):
            await update_mastery(db_session, "derivative_basic", is_correct=True, user_id=1)
        mastery = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        assert mastery.total_attempts == 5
        assert mastery.correct_attempts == 5
        assert mastery.p_mastery > 0.5


# ━━ get_all_masteries() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetAllMasteries:

    @pytest.mark.asyncio
    async def test_empty_db(self, db_session):
        masteries = await get_all_masteries(db_session, user_id=99)
        assert masteries == {}

    @pytest.mark.asyncio
    async def test_returns_dict(self, db_session):
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        await get_or_create_mastery(db_session, "integral_definite", user_id=1)
        masteries = await get_all_masteries(db_session, user_id=1)
        assert isinstance(masteries, dict)
        assert "derivative_basic" in masteries
        assert "integral_definite" in masteries

    @pytest.mark.asyncio
    async def test_values_are_floats(self, db_session):
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        masteries = await get_all_masteries(db_session, user_id=1)
        for v in masteries.values():
            assert isinstance(v, float)


# ━━ get_mastery_profile() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetMasteryProfile:

    @pytest.mark.asyncio
    async def test_empty_db(self, db_session):
        profile = await get_mastery_profile(db_session, user_id=99)
        assert profile == []

    @pytest.mark.asyncio
    async def test_profile_structure(self, db_session):
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        profile = await get_mastery_profile(db_session, user_id=1)
        assert len(profile) == 1
        entry = profile[0]
        assert "skill_id" in entry
        assert "skill_name" in entry
        assert "p_mastery" in entry
        assert "level" in entry
        assert "total_attempts" in entry
        assert "correct_attempts" in entry

    @pytest.mark.asyncio
    async def test_level_is_string(self, db_session):
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        profile = await get_mastery_profile(db_session, user_id=1)
        assert profile[0]["level"] in ["beginner", "developing", "proficient", "mastered"]


# ━━ identify_gaps() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIdentifyGaps:

    @pytest.mark.asyncio
    async def test_no_gaps_when_all_mastered(self, db_session):
        m = await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        m.p_mastery = 0.9
        await db_session.flush()

        gaps = await identify_gaps(db_session, "derivative_rules", user_id=1)
        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_finds_weak_prerequisites(self, db_session):
        # derivative_basic is weak (default 0.1)
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        gaps = await identify_gaps(db_session, "derivative_rules", user_id=1)
        assert len(gaps) > 0
        assert any(g["skill_id"] == "derivative_basic" for g in gaps)

    @pytest.mark.asyncio
    async def test_gap_structure(self, db_session):
        await get_or_create_mastery(db_session, "derivative_basic", user_id=1)
        gaps = await identify_gaps(db_session, "derivative_rules", user_id=1)
        if gaps:
            g = gaps[0]
            assert "skill_id" in g
            assert "skill_name" in g
            assert "current_mastery" in g
            assert "recommendation" in g

    @pytest.mark.asyncio
    async def test_no_prerequisites_no_gaps(self, db_session):
        gaps = await identify_gaps(db_session, "derivative_basic", user_id=1)
        assert gaps == []


# ━━ get_recent_results() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetRecentResults:

    @pytest.mark.asyncio
    async def test_empty_results(self, db_session):
        results = await get_recent_results(db_session, "derivative_basic", user_id=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_list_of_bools(self, db_session):
        # Add some interaction logs
        for correct in [True, False, True]:
            log = InteractionLog(
                user_id=1,
                question="test",
                skill_id="derivative_basic",
                is_correct=correct,
            )
            db_session.add(log)
        await db_session.flush()

        results = await get_recent_results(db_session, "derivative_basic", user_id=1)
        assert len(results) == 3
        assert all(isinstance(r, bool) for r in results)

    @pytest.mark.asyncio
    async def test_respects_limit(self, db_session):
        for i in range(15):
            log = InteractionLog(
                user_id=1,
                question=f"q{i}",
                skill_id="derivative_basic",
                is_correct=True,
            )
            db_session.add(log)
        await db_session.flush()

        results = await get_recent_results(db_session, "derivative_basic", user_id=1, limit=5)
        assert len(results) <= 5
