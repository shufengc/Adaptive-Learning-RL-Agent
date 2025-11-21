from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StudentState:
    # Container for the simulated student state S_t, including mastery, fatigue, attention, forgetting factor, and last-question difficulty.
    mastery: float
    fatigue: float
    attention: float
    forget_rate: float
    last_difficulty: float
    
    # Return the student state as a NumPy vector for RL algorithms.
    def as_array(self) -> np.ndarray:
        """Return state as a 1D numpy array, convenient for RL."""
        return np.array(
            [
                self.mastery,
                self.fatigue,
                self.attention,
                self.forget_rate,
                self.last_difficulty,
            ],
            dtype=np.float32,
        )
    
    # Return the state as a pure Python tuple when a lightweight representation is required.
    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        """Sometimes it is useful to have a pure python tuple."""
        return (
            float(self.mastery),
            float(self.fatigue),
            float(self.attention),
            float(self.forget_rate),
            float(self.last_difficulty),
        )

# Initialize the starting state S_0 with configurable defaults.
def init_student_state(
    init_mastery: float = 0.5,
    init_fatigue: float = 0.0,
    init_attention: float = 1.0,
    # Initial forgetting factor is set to 1.0 (i.e., no decay applied at t=0).
    forget_rate: float = 1.0,          
    init_last_difficulty: float = 0.5,
) -> StudentState:
    """
    Factory function used by the environment to initialize S_0.
    These defaults can be changed later if the group decides so.
    """
    return StudentState(
        mastery=init_mastery,
        fatigue=init_fatigue,
        attention=init_attention,
        forget_rate=forget_rate,
        last_difficulty=init_last_difficulty,
    )
