# AI Teacher Strategy Project (Reinforcement Learning)

This project aims to build an **AI Teacher** capable of selecting personalized question difficulty levels for each student, based on the student’s real-time learning state. The pipeline consists of four phases.

---

## Phase 1 — Data Preparation & Feature Engineering

### Objectives
Clean raw student–question interaction data and extract all features needed for student modeling and RL training.

### Key Components
- **Δt (time gap):** Millisecond-level difference between consecutive attempts.
- **Question difficulty (D):** Computed as historical error rate

  `D_t = 1 - (# correct answers / # total attempts)`

### Outputs
- Cleaned sequence data (train + test)
- Step-level dataset for parameter estimation
- Per-question difficulty mapping

---

## Phase 2 — Student Simulator & Parameter Fitting

### Objective
Fit population-level cognitive dynamics parameters and build a differentiable virtual student model.

### Fitted Parameters
- **β** — fatigue accumulation rate  
- **γ** — attention decay rate  
- **λ** — forgetting rate  
- **lr** — learning rate

### Model Mechanics

Student state:

`S_t = [M_t, F_t, A_t, Forget_t, D_t_last]`

State transition equations:

- **Attention**  
  `A_{t+1} = exp(-γ * Δt)`

- **Forgetting factor**  
  `Forget = exp(-λ * Δt)`

- **Mastery update**  
  - If correct: `M_new = M_t + lr * (1 - M_t)`  
  - If wrong:   `M_new = M_t - 0.2 * lr`  
  - Final:      `M_{t+1} = M_new * Forget`

- **Fatigue**  
  `F_{t+1} = min(1, F_t + β)`

Student correctness model:

`P(correct) = mastery * attention * (1 - fatigue) * (1 - 0.1 * difficulty)`

### Deliverables
- `fit_student_params.py` — parameter estimation
- `environment.py`, `student_model.py`, `student_state.py` — student simulator
- `fitted_params.json` — fitted global parameters

### Evaluation
- AUC on unseen student sequences  
- Typical result: **AUC ≈ 0.72**

---

## Phase 3 — Reinforcement Learning Training (DQN)

### Objective
Train an RL agent to select the next question difficulty level that maximizes long-term learning outcomes.

### RL Formulation
- **Action:** next difficulty category
- **Cognitive load:** `C_t = difficulty * (1 - mastery)`
- **Reward:**  
  `R_t = ΔM_t - α * C_t - β * F_t + γ * A_t`

### Output
Trained DQN agent capable of adaptive teaching.

*(RL code and training loop belong to later phases and are not part of this repository yet.)*

---

## Phase 4 — Final Strategy Evaluation & Visualization

### Objectives
Evaluate the long-term performance of the AI teacher in comparison with baseline strategies.

### Metrics
- Mastery improvement
- Fatigue control
- Cognitive load reduction
- Attention sustainability

### Visualization
Time-series plots of:
- Mastery trajectory
- Fatigue level
- Attention profile
- Question difficulty schedule

---

## Repository Structure

```text
project/
  data/
    train_valid_sequences_quelevel_clean_time.xlsx
    train_valid_sequences_question_stats.csv
    test_quelevel_cleaned_time.csv
    fitted_params.json
  env/
    __init__.py
    environment.py
    student_model.py
    student_state.py
    run_demo.py
  fit_student_params.py
  evaluate_auc.py
  README.md
```

---

## How to Run

### 1. Fit parameters
```bash
python fit_student_params.py
```

### 2. Evaluate AUC
```bash
python evaluate_auc.py
```

### 3. (Optional) Run virtual student demo
```bash
python -m env.run_demo
```

---

## Requirements
- Python 3.9+
- NumPy
- pandas

---

## Notes
- All scripts run offline and require no third-party ML libraries.
- The virtual student environment (`env/`) is essential for Phases 2–4.
- The RL training pipeline (Phase 3) will build on this environment.
