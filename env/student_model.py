import math
from typing import Optional

import numpy as np

from .student_state import StudentState

_GLOBAL_RNG = np.random.default_rng()

# Compute the probability of answering correctly based on mastery, attention, fatigue, and difficulty.
def prob_correct(state: StudentState, difficulty: float) -> float:
    """
    P(correct) = mastery * attention * (1 - fatigue) * (1 - 0.1 * difficulty)
    """
    p = (
        state.mastery
        * state.attention
        * (1.0 - state.fatigue)
        * (1.0 - 0.1 * difficulty)
    )
    return float(np.clip(p, 0.01, 0.99))

# Sample a binary response (correct/incorrect) given the current state and question difficulty.
def sample_response(
    state: StudentState,
    difficulty: float,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample a binary response given current state and difficulty.
    return 1 if correct, 0 otherwise.
    """
    if rng is None:
        rng = _GLOBAL_RNG

    p = prob_correct(state, difficulty)
    return int(rng.random() < p)

# Update the student state from S_t to S_{t+1} according to the model’s transition rules.
def update_state(
    state: StudentState,
    correct: int,
    difficulty: float,
    delta_t: float = 1.0,
    lr: float = 0.05,     # lr≈0.05
    lam: float = 0.001,   # λ：forget
    beta: float = 0.01,   # β：fatigue accumulation
    gamma: float = 0.001, # γ：attentio
) -> StudentState:
    # Mastery update:
    # If the student answers correctly:
    # M_new = M_t + lr * (1 - M_t)
    # If incorrect:
    # M_new = M_t - lr * 0.2
    # Apply exponential forgetting over the time gap:
    # ForgetFactor = exp(-lam * Δt)
    # M_{t+1} = M_new * ForgetFactor
    if correct == 1:
        mastery_new = state.mastery + lr * (1.0 - state.mastery)
    else:
        mastery_new = state.mastery - lr * 0.2

    mastery_new = max(0.0, min(1.0, mastery_new))

    forget_factor = math.exp(-lam * delta_t)
    mastery_next = mastery_new * forget_factor
    mastery_next = max(0.0, min(1.0, mastery_next))

    # Fatigue update:
    # F_{t+1} = min(1.0, F_t + beta)
    fatigue_next = min(1.0, state.fatigue + beta)

    # Attention update:
    # A_{t+1} = exp(-gamma * Δt)
    attention_next = math.exp(-gamma * delta_t)
    
    # Store the current forgetting factor for use in downstream state representations.
    return StudentState(
        mastery=mastery_next,
        fatigue=fatigue_next,
        attention=attention_next,
        forget_rate=forget_factor,
        last_difficulty=difficulty,
    )
