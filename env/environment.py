import os
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from .student_state import StudentState, init_student_state
from .student_model import prob_correct, update_state

# Resolve the absolute path of the data directory.
def _get_data_dir() -> str:
    here = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(here, "..", "data"))
    return data_dir

# Load per-question accuracy statistics and convert accuracy into difficulty scores.
def load_question_difficulty(
    filename: str = "train_valid_sequences_question_stats.csv",
) -> Dict[int, float]:
    data_dir = _get_data_dir()
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Question stats file not found: {path}")

    df = pd.read_csv(path)

    # Ensure required columns exist in the statistics file.
    if "question" not in df.columns or "accuracy" not in df.columns:
        raise ValueError(
            "Expected columns 'question' and 'accuracy' "
            "in train_valid_sequences_question_stats.csv"
        )

    df["difficulty"] = (1.0 - df["accuracy"]).clip(0.05, 0.95)
    difficulty_map: Dict[int, float] = (
        df.set_index("question")["difficulty"].astype(float).to_dict()
    )
    return difficulty_map

# A simplified RL-style environment that simulates a student's response behavior.
class StudentEnv:

    def __init__(
        self,
        question2difficulty: Optional[Dict[int, float]] = None,
        max_steps: int = 50,
        rng_seed: Optional[int] = None,
        fixed_delta_t: float = 1.0,
        lr: float = 0.05,
        lam: float = 0.001,
        beta: float = 0.01,
        gamma: float = 0.001,
    ) -> None:
        """
        Args:
            question2difficulty: optional precomputed difficulty map;
                                 if None, it will be loaded from data/.
            max_steps          : maximum number of steps in an episode.
            rng_seed           : random seed for reproducibility.
            fixed_delta_t      : Δt used in the simulator.
            lr, lam, beta,
            gamma              : dynamical parameters of the student model.
        """
        if question2difficulty is None:
            question2difficulty = load_question_difficulty()

        self.question2difficulty: Dict[int, float] = question2difficulty
        self.questions: np.ndarray = np.array(
            list(self.question2difficulty.keys()), dtype=np.int64
        )
        self.num_actions: int = self.questions.shape[0]
        self.max_steps: int = max_steps

        self.rng = np.random.default_rng(rng_seed)

        # Dynamics parameters governing state transitions.
        self.delta_t = float(fixed_delta_t)
        self.lr = float(lr)
        self.lam = float(lam)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # Internal environment state and step counter used during simulation.
        self.state: Optional[StudentState] = None
        self.steps_done: int = 0

    # Reset the environment and return the initial student state vector.
    def reset(self) -> np.ndarray:
        self.state = init_student_state()
        self.steps_done = 0
        return self.state.as_array()
    
    # Execute a single environment step given an action (question index).
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Args:
            action: integer in [0, num_actions - 1], indexes into the question list.

        Returns:
            obs   : next state vector (np.ndarray)
            reward: float, here simply 1.0 if correct else 0.0
            done  : bool, whether the episode has terminated
            info  : dict with extra information (question_id, difficulty, correct, etc.)
        """
        if self.state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if not (0 <= action < self.num_actions):
            raise ValueError(
                f"Invalid action {action}. "
                f"Must be in [0, {self.num_actions - 1}]."
            )

        # Map action index to the corresponding question ID and its difficulty.
        question_id = int(self.questions[action])
        difficulty = float(self.question2difficulty[question_id])

        # Sample correctness of the response using the current student state.
        p = prob_correct(self.state, difficulty)
        correct = int(self.rng.random() < p)
        reward = float(correct)

        # Update the student state using the model’s dynamics.
        next_state = update_state(
            state=self.state,
            correct=correct,
            difficulty=difficulty,
            delta_t=self.delta_t,
            lr=self.lr,
            lam=self.lam,
            beta=self.beta,
            gamma=self.gamma,
        )

        # Check whether the episode has reached the maximum number of steps.
        self.state = next_state
        self.steps_done += 1
        done = self.steps_done >= self.max_steps
        
        # Package additional diagnostic information for logging and debugging.
        obs = next_state.as_array()
        info: Dict[str, Any] = {
            "question_id": question_id,
            "difficulty": difficulty,
            "correct": correct,
            "prob_correct": p,
            "step": self.steps_done,
        }
        return obs, reward, done, info

# Convenience helper to construct an environment instance with default settings.
def make_default_env(
    max_steps: int = 50,
    rng_seed: Optional[int] = None,
    fixed_delta_t: float = 1.0,
    lr: float = 0.05,
    lam: float = 0.001,
    beta: float = 0.01,
    gamma: float = 0.001,
) -> StudentEnv:
    q2d = load_question_difficulty()
    return StudentEnv(
        question2difficulty=q2d,
        max_steps=max_steps,
        rng_seed=rng_seed,
        fixed_delta_t=fixed_delta_t,
        lr=lr,
        lam=lam,
        beta=beta,
        gamma=gamma,
    )
