"""Tests for ExerciseType.parse."""

import pytest

from app.models import ExerciseType


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Squat", ExerciseType.SQUAT),
        ("squat", ExerciseType.SQUAT),
        ("  SQUAT ", ExerciseType.SQUAT),
        ("Push-up", ExerciseType.PUSH_UP),
        ("push_up", ExerciseType.PUSH_UP),
        ("pushup", ExerciseType.PUSH_UP),
        ("t bar row", ExerciseType.T_BAR_ROW),
        ("ROMANIAN_DEADLIFT", ExerciseType.ROMANIAN_DEADLIFT),
    ],
)
def test_parse_accepts_common_spellings(raw, expected):
    assert ExerciseType.parse(raw) is expected


@pytest.mark.parametrize("raw", ["", "burpee", "squatt", "bench"])
def test_parse_rejects_unknown(raw):
    with pytest.raises(ValueError):
        ExerciseType.parse(raw)


def test_every_member_round_trips():
    for member in ExerciseType:
        assert ExerciseType.parse(member.value) is member
        assert ExerciseType.parse(member.name) is member
