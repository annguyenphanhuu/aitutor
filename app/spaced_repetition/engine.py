"""SM-2 Spaced Repetition Engine — implements the SuperMemo SM-2 algorithm."""

from datetime import datetime, timedelta


def sm2_update(
    easiness: float,
    interval: int,
    repetitions: int,
    quality: int,
) -> dict:
    """
    Apply the SM-2 algorithm to update spaced repetition parameters.

    Args:
        easiness:    Current easiness factor (≥ 1.3)
        interval:    Current interval in days
        repetitions: Number of consecutive correct reviews
        quality:     Student self-rating 0-5:
                     0 = complete blank
                     1 = wrong, but recognized after seeing answer
                     2 = wrong, but answer seemed easy to recall
                     3 = correct with serious difficulty
                     4 = correct after hesitation
                     5 = perfect, instant recall

    Returns:
        dict with: easiness, interval, repetitions, next_review (datetime)
    """
    # Update easiness factor
    new_ef = easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    if quality >= 3:
        # Correct response
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = round(interval * new_ef)
        new_reps = repetitions + 1
    else:
        # Incorrect response — reset
        new_interval = 1
        new_reps = 0

    next_review = datetime.utcnow() + timedelta(days=new_interval)

    return {
        "easiness": round(new_ef, 2),
        "interval": new_interval,
        "repetitions": new_reps,
        "next_review": next_review,
    }


def quality_from_quiz(is_correct: bool, time_seconds: float = None) -> int:
    """
    Convert a quiz result to an SM-2 quality rating.

    Args:
        is_correct: Whether the student answered correctly
        time_seconds: How long the student took (optional)

    Returns:
        Quality rating 0-5
    """
    if not is_correct:
        return 1  # Wrong but saw the answer

    if time_seconds is not None:
        if time_seconds < 10:
            return 5  # Perfect instant recall
        elif time_seconds < 30:
            return 4  # Correct after brief thought
        else:
            return 3  # Correct with difficulty
    else:
        return 4  # Default for correct answer
