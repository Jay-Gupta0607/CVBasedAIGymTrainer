"""Per-exercise form rules, as data.

Every `ExerciseType` has a profile. A profile says which joints matter (`primary`),
what range of motion a repetition should reach (`rom`), and hard frame-level bounds
(`limits`).

The numbers are general-purpose HEURISTIC DEFAULTS chosen to be lenient - they are not
clinical standards and have not been tuned against labelled footage. They assume the
angles from `pose_features.compute_joint_angles` (0 = folded, 180 = straight; torso
lean 0 = upright). Adjust them here; nothing else hard-codes thresholds.

Joint names are side-agnostic ("knee" covers left_knee/right_knee): a joint's value is
the mean of the sides that were measured, so side-on videos (one side visible) work.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from app.models import ExerciseType

SIDED_JOINTS = ("elbow", "shoulder", "knee", "hip")


@dataclass(frozen=True)
class RomTarget:
    """A repetition must reach `threshold` at its `extreme` ("min" or "max") of `joint`."""

    joint: str
    extreme: str  # "min": angle must get down to threshold; "max": must get up to it
    threshold: float
    issue: str


@dataclass(frozen=True)
class LimitRule:
    """Frame-level bound: `joint` should stay within [low, high]."""

    joint: str
    low: Optional[float] = None
    high: Optional[float] = None
    issue_low: str = ""
    issue_high: str = ""


@dataclass(frozen=True)
class ExerciseProfile:
    primary: Tuple[str, ...] = ()
    rom: Tuple[RomTarget, ...] = ()
    limits: Tuple[LimitRule, ...] = ()

    @property
    def rep_signal(self) -> Optional[RomTarget]:
        """The ROM target whose joint cycles once per repetition (the first listed)."""
        return self.rom[0] if self.rom else None


def _rom(joint: str, extreme: str, threshold: float, issue: str) -> RomTarget:
    return RomTarget(joint, extreme, threshold, issue)


_CURL = ExerciseProfile(
    primary=("elbow",),
    rom=(
        _rom("elbow", "min", 65, "Curl not reaching full contraction"),
        _rom("elbow", "max", 150, "Arms not fully extending at the bottom"),
    ),
    limits=(LimitRule("torso_lean", high=20, issue_high="Leaning back / swinging the weight"),),
)

_BENCH = ExerciseProfile(
    primary=("elbow",),
    rom=(
        _rom("elbow", "min", 100, "Bar not lowered far enough"),
        _rom("elbow", "max", 155, "Not locking out at the top"),
    ),
)

_ROW_PULL = ExerciseProfile(
    primary=("elbow", "shoulder"),
    rom=(
        _rom("elbow", "min", 100, "Bar not pulled down far enough"),
        _rom("elbow", "max", 150, "Arms not fully extending at the top"),
    ),
    limits=(LimitRule("torso_lean", high=35, issue_high="Leaning back too far"),),
)

EXERCISE_PROFILES: Mapping[ExerciseType, ExerciseProfile] = {
    ExerciseType.BARBELL_BICEPS_CURL: _CURL,
    ExerciseType.HAMMER_CURL: _CURL,
    ExerciseType.BENCH_PRESS: _BENCH,
    ExerciseType.INCLINE_BENCH_PRESS: _BENCH,
    ExerciseType.DECLINE_BENCH_PRESS: _BENCH,
    # Comparison-only: no reliable absolute target from a single camera angle
    ExerciseType.CHEST_FLY_MACHINE: ExerciseProfile(primary=("shoulder", "elbow")),
    ExerciseType.RUSSIAN_TWIST: ExerciseProfile(primary=("hip",)),
    ExerciseType.DEADLIFT: ExerciseProfile(
        primary=("hip", "knee"),
        rom=(_rom("hip", "max", 165, "Not reaching full lockout at the top"),),
    ),
    ExerciseType.ROMANIAN_DEADLIFT: ExerciseProfile(
        primary=("hip", "knee"),
        rom=(
            _rom("hip", "max", 160, "Not reaching full lockout at the top"),
            _rom("hip", "min", 110, "Not hinging deep enough"),
        ),
        limits=(LimitRule("knee", low=130, issue_low="Knees bending too much (keep them soft, not squatting)"),),
    ),
    ExerciseType.HIP_THRUST: ExerciseProfile(
        primary=("hip",),
        rom=(_rom("hip", "max", 165, "Not reaching full hip extension"),),
    ),
    ExerciseType.LAT_PULLDOWN: _ROW_PULL,
    ExerciseType.LATERAL_RAISE: ExerciseProfile(
        primary=("shoulder",),
        rom=(_rom("shoulder", "max", 75, "Arms not reaching shoulder height"),),
        limits=(
            LimitRule("elbow", low=110, issue_low="Elbows excessively bent"),
            LimitRule("torso_lean", high=20, issue_high="Leaning / swinging the torso"),
        ),
    ),
    ExerciseType.LEG_EXTENSION: ExerciseProfile(
        primary=("knee",),
        rom=(_rom("knee", "max", 160, "Not fully extending the knees"),),
    ),
    ExerciseType.LEG_RAISES: ExerciseProfile(
        primary=("hip",),
        rom=(_rom("hip", "min", 100, "Legs not raised high enough"),),
    ),
    ExerciseType.PLANK: ExerciseProfile(
        primary=("hip",),
        limits=(LimitRule("hip", low=155, issue_low="Hips sagging or piking"),),
    ),
    ExerciseType.PULL_UP: ExerciseProfile(
        primary=("elbow", "shoulder"),
        rom=(
            _rom("elbow", "min", 75, "Chin not reaching the bar"),
            _rom("elbow", "max", 155, "Not reaching a full hang"),
        ),
    ),
    ExerciseType.PUSH_UP: ExerciseProfile(
        primary=("elbow",),
        rom=(
            _rom("elbow", "min", 100, "Not lowering the chest far enough"),
            _rom("elbow", "max", 155, "Not locking out at the top"),
        ),
        limits=(LimitRule("hip", low=155, issue_low="Hips sagging or piking"),),
    ),
    ExerciseType.SHOULDER_PRESS: ExerciseProfile(
        primary=("elbow", "shoulder"),
        rom=(
            _rom("elbow", "max", 160, "Not locking out overhead"),
            _rom("elbow", "min", 105, "Not lowering the weight far enough"),
        ),
        limits=(LimitRule("torso_lean", high=20, issue_high="Excessive back lean"),),
    ),
    ExerciseType.SQUAT: ExerciseProfile(
        primary=("knee", "hip"),
        rom=(
            _rom("knee", "min", 100, "Insufficient depth"),
            _rom("knee", "max", 160, "Not standing fully at the top"),
        ),
        limits=(LimitRule("torso_lean", high=60, issue_high="Excessive forward lean"),),
    ),
    ExerciseType.T_BAR_ROW: ExerciseProfile(
        primary=("elbow",),
        rom=(
            _rom("elbow", "min", 95, "Not pulling the weight high enough"),
            _rom("elbow", "max", 150, "Arms not fully extending at the bottom"),
        ),
    ),
    ExerciseType.TRICEP_DIPS: ExerciseProfile(
        primary=("elbow",),
        rom=(
            _rom("elbow", "min", 100, "Not lowering deep enough"),
            _rom("elbow", "max", 155, "Not locking out at the top"),
        ),
    ),
    ExerciseType.TRICEP_PUSHDOWN: ExerciseProfile(
        primary=("elbow",),
        rom=(_rom("elbow", "max", 155, "Not fully extending the arms"),),
        limits=(LimitRule("torso_lean", high=25, issue_high="Leaning over the weight"),),
    ),
}

# What a signed difference (user - trainer) means, per joint: (user smaller, user larger)
DEVIATION_TEXT: Dict[str, Tuple[str, str]] = {
    "knee": ("Knees more bent than the trainer", "Knees straighter than the trainer"),
    "hip": ("Hips more flexed than the trainer", "Hips more extended than the trainer"),
    "elbow": ("Elbows more bent than the trainer", "Elbows straighter than the trainer"),
    "shoulder": ("Arms held lower than the trainer", "Arms raised higher than the trainer"),
    "torso_lean": ("Torso more upright than the trainer", "Torso leaning more than the trainer"),
}

_EMPTY_PROFILE = ExerciseProfile()


def get_profile(exercise_name: str) -> ExerciseProfile:
    """Profile for an exercise name in any common spelling ("squat", "Push-up", ...).

    Unknown names get an empty profile (no absolute rules), never an error.
    """
    try:
        return EXERCISE_PROFILES.get(ExerciseType.parse(exercise_name), _EMPTY_PROFILE)
    except ValueError:
        return _EMPTY_PROFILE


def joint_value(angles: Mapping[str, float], joint: str) -> Optional[float]:
    """Mean of the measured sides of `joint`, or the value itself for `torso_lean`."""
    if joint in SIDED_JOINTS:
        values = [angles[k] for k in (f"left_{joint}", f"right_{joint}") if k in angles]
        return float(sum(values) / len(values)) if values else None
    return angles.get(joint)
