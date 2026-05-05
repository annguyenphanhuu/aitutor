"""Bayesian Knowledge Tracing (BKT) implementation."""


class BKTModel:
    """
    Bayesian Knowledge Tracing estimates the probability that
    a student has mastered a particular skill.

    Parameters per skill:
    - p_init:    P(L₀)  — probability of knowing the skill initially
    - p_transit: P(T)   — probability of learning the skill after practice
    - p_slip:   P(S)   — probability of answering wrong despite mastery
    - p_guess:  P(G)   — probability of answering correctly by chance
    """

    # Default BKT parameters (can be tuned per skill)
    DEFAULT_PARAMS = {
        "p_init": 0.1,
        "p_transit": 0.1,
        "p_slip": 0.05,
        "p_guess": 0.25,
    }

    def __init__(self, params: dict = None):
        p = params or self.DEFAULT_PARAMS
        self.p_init = p.get("p_init", 0.1)
        self.p_transit = p.get("p_transit", 0.1)
        self.p_slip = p.get("p_slip", 0.05)
        self.p_guess = p.get("p_guess", 0.25)

    def update(self, p_mastery: float, is_correct: bool) -> float:
        """
        Update mastery probability after observing a response.

        Returns the new P(mastery).
        """
        if is_correct:
            # P(L|correct) = P(correct|L) * P(L) / P(correct)
            p_correct_given_mastery = 1.0 - self.p_slip
            p_correct_given_not = self.p_guess
            p_correct = p_correct_given_mastery * p_mastery + p_correct_given_not * (1 - p_mastery)

            if p_correct > 0:
                p_mastery_posterior = (p_correct_given_mastery * p_mastery) / p_correct
            else:
                p_mastery_posterior = p_mastery
        else:
            # P(L|incorrect) = P(incorrect|L) * P(L) / P(incorrect)
            p_incorrect_given_mastery = self.p_slip
            p_incorrect_given_not = 1.0 - self.p_guess
            p_incorrect = p_incorrect_given_mastery * p_mastery + p_incorrect_given_not * (1 - p_mastery)

            if p_incorrect > 0:
                p_mastery_posterior = (p_incorrect_given_mastery * p_mastery) / p_incorrect
            else:
                p_mastery_posterior = p_mastery

        # Apply learning transition: P(L_new) = P(L_posterior) + (1 - P(L_posterior)) * P(T)
        p_mastery_new = p_mastery_posterior + (1 - p_mastery_posterior) * self.p_transit

        return max(0.0, min(1.0, p_mastery_new))

    def predict_correct(self, p_mastery: float) -> float:
        """Predict probability of a correct answer given current mastery."""
        return p_mastery * (1 - self.p_slip) + (1 - p_mastery) * self.p_guess

    def get_mastery_level(self, p_mastery: float) -> str:
        """Classify mastery into human-readable levels."""
        if p_mastery < 0.3:
            return "beginner"
        elif p_mastery < 0.6:
            return "developing"
        elif p_mastery < 0.85:
            return "proficient"
        else:
            return "mastered"
