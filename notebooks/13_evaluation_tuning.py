# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 13. Evaluation tuning and playing strength
#
# **Learning goals:** frame evaluation weights as measurable parameters, understand feature ablation and labeled-position tuning, calculate a small logistic/Texel-style objective, and design matches that report uncertainty rather than overclaiming.
#
# Lesson 05 built material, piece-square, pawn, and mobility features. This lesson shows how to evaluate and tune such terms; the engine exposes `basic` and `positional` profiles for ablation. The toy parameter demonstration is not wired into live search or its evaluator weights.
#
# Sources: [CPW Evaluation](https://www.chessprogramming.org/Evaluation), [CPW Texel's Tuning Method](https://www.chessprogramming.org/Texel%27s_Tuning_Method), [CPW Playing Strength](https://www.chessprogramming.org/Playing_Strength)
# %%
import math


# %% [markdown]
# ## Start with feature ablation
#
# A feature is a measurable term such as material balance, king exposure, or pawn structure. Before tuning its coefficient, run an **ablation**: compare the baseline with the feature disabled while keeping positions, search settings, and everything else fixed. Inspect score changes and tactical/strategic examples; then use matches to ask whether it helps decisions. A feature can be correlated with another, incorrectly signed, too expensive, or useful only in a phase. Removing one term can also expose interactions, so ablation is diagnosis, not proof that terms are independent.
#
# A labeled-position data set pairs positions with outcomes. For White-perspective score `s` in centipawns, a logistic model predicts a win-like score `p = 1 / (1 + 10 ** (-s / scale))`. Given target outcome `y` in [0, 1] (win=1, draw=0.5, loss=0), a common cross-entropy/Texel-style objective penalizes prediction error. The label is an observed result, not a perfect measure of position truth: player strength, time control, opening choice, and sampling all affect it.
# %%
def predicted_score(score_cp: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-score_cp / scale))


def mean_log_loss(scores: list[float], outcomes: list[float], scale: float) -> float:
    if len(scores) != len(outcomes) or not scores:
        raise ValueError('scores and outcomes must be non-empty and equally sized')
    epsilon = 1e-12
    losses = []
    for score, outcome in zip(scores, outcomes, strict=True):
        if not 0.0 <= outcome <= 1.0:
            raise ValueError('outcomes must be in [0, 1]')
        p = min(1.0 - epsilon, max(epsilon, predicted_score(score, scale)))
        losses.append(-(outcome * math.log(p) + (1.0 - outcome) * math.log(1.0 - p)))
    return sum(losses) / len(losses)


# Tiny synthetic set: a won position, an even draw, and a lost position.
# Two candidate parameter settings yield different predictions/objectives.
labels = [1.0, 0.5, 0.0]
feature_values = [1.0, 0.0, -1.0]
weight_candidates = [40.0, 120.0]  # cp per toy feature unit
for weight in weight_candidates:
    scores = [weight * value for value in feature_values]
    print(f'weight={weight:5.1f} cp/unit -> scores={scores}, log loss={mean_log_loss(scores, labels, 200):.4f}')
assert mean_log_loss([120, 0, -120], labels, 200) < mean_log_loss([40, 0, -40], labels, 200)
assert predicted_score(0, 200) == 0.5

# %% [markdown]
# ## A tiny deterministic numerical tuning step
#
# The objective is differentiable. For the base-10 logistic above, the per-sample derivative with respect to predicted score in centipawns includes `ln(10) / scale`; chaining through `score = weight * feature` gives a gradient for the weight. One small gradient-descent update illustrates the mechanics. Real tuning uses many positions, careful scaling, stopping/regularization choices, and validation; this toy data is far too small to learn a chess parameter.
# %%

def weight_gradient(weight: float, features: list[float], outcomes: list[float], scale: float) -> float:
    if len(features) != len(outcomes) or not features:
        raise ValueError('features and outcomes must be non-empty and equally sized')
    coefficient = math.log(10.0) / scale
    total = 0.0
    for feature, outcome in zip(features, outcomes, strict=True):
        probability = predicted_score(weight * feature, scale)
        total += (probability - outcome) * coefficient * feature
    return total / len(features)


initial_weight = 100.0
learning_rate = 50.0
gradient = weight_gradient(initial_weight, feature_values, labels, scale=200)
updated_weight = initial_weight - learning_rate * gradient

assert math.isfinite(gradient) and updated_weight > initial_weight
print(f'one reproducible gradient step: {initial_weight:.1f} -> {updated_weight:.3f} cp/unit')

# %% [markdown]
# ## Split data and respect the labels
#
# Split by game (or opening/player/time-control group), not randomly by individual adjacent positions: positions from one game are highly correlated and leakage makes held-out results look better than they are. Keep a validation set untouched while choosing parameters; if repeatedly tuning against it, it has become training data and a new holdout is needed.
#
# Draws matter. They are common in strong chess and sit at outcome 0.5, not a win or loss. If the corpus is draw-heavy, report outcome frequencies and inspect calibration by outcome/phase; indiscriminately balancing classes changes the target distribution. Filtering, weighting, or sampling can be legitimate but must be explicit, because it changes what the objective means.
#
# Avoid learning the dataset's quirks: a large training improvement with worse validation behavior is overfitting. Use fixed, documented game sources and remove duplicates or near-duplicate positions across partitions.
#
# ## Playing-strength measurement is a separate experiment
#
# Use paired color-swapped games, fixed openings or balanced opening suites, equal time controls, identical hardware/search budgets, and pinned engine versions/settings. Alternate colors and opening order to reduce first-move and opening bias. Keep adjudication, draw handling, crashes, and time forfeits explicit; do not silently discard inconvenient results. Self-play is useful for relative regression testing but can reward shared blind spots and does not estimate absolute human strength.
#
# Elo is a model-based summary of expected score, not a direct measurement of absolute playing quality. Small match samples have wide uncertainty; report games, wins/draws/losses, conditions, rating estimate, and an interval or uncertainty method. Results depend on opponent pool, openings, time control, hardware, and version. A tiny sample cannot establish that a parameter set or engine is stronger.
# %% [markdown]
# ## A trinomial SPRT example
#
# A sequential probability ratio test (SPRT) compares two stated outcome models. The probabilities below are an **illustrative model**, not evidence about either engine's strength. Under H0, win/draw/loss probabilities are (0.30, 0.40, 0.30); under H1 they are (0.40, 0.40, 0.20). Each model sums to one. Draw is its own category in the likelihood: do not split it into half-wins and half-losses.
#
# For each game, add `log(P(outcome | H1) / P(outcome | H0))`. With alpha=beta=0.05, Wald's upper and lower log-likelihood boundaries are `log((1-beta)/alpha)` and `log(beta/(1-alpha))`. Crossing the upper boundary favors H1; crossing the lower favors H0; otherwise continue. This compact model assumes independent games from the same probabilities. Real match design still needs paired colors/openings and uncertainty reporting as above. The repository's `stockfish.play_match`/`run_series` infrastructure can run matches, but this example uses only the standard library and needs no Stockfish binary.
# %%


H0 = {'win': 0.30, 'draw': 0.40, 'loss': 0.30}
H1 = {'win': 0.40, 'draw': 0.40, 'loss': 0.20}
assert math.isclose(sum(H0.values()), 1.0)
assert math.isclose(sum(H1.values()), 1.0)

alpha = beta = 0.05
upper_boundary = math.log((1 - beta) / alpha)
lower_boundary = math.log(beta / (1 - alpha))
log_increment = {outcome: math.log(H1[outcome] / H0[outcome]) for outcome in H0}
outcomes = ['win'] * 10 + ['draw', 'loss', 'win', 'win']
cumulative_log_likelihood = 0.0
decision = 'continue'
decision_game = None
for game, outcome in enumerate(outcomes, start=1):
    cumulative_log_likelihood += log_increment[outcome]
    if cumulative_log_likelihood >= upper_boundary:
        decision, decision_game = 'favor H1', game
        break
    if cumulative_log_likelihood <= lower_boundary:
        decision, decision_game = 'favor H0', game
        break

assert math.isclose(upper_boundary, math.log(19))
assert math.isclose(lower_boundary, -math.log(19))
assert math.isclose(log_increment['draw'], 0.0)
assert math.isclose(10 * log_increment['win'], 2.8768207245, rel_tol=1e-9)
assert 10 * log_increment['win'] < upper_boundary < 11 * log_increment['win']
assert set(outcomes) == set(H0)
assert (decision, decision_game) == ('favor H1', 14)
print(f'upper={upper_boundary:.3f}, lower={lower_boundary:.3f}; '
      f'{decision} after {decision_game} games '
      f'(log-likelihood={cumulative_log_likelihood:.3f})')

# %% [markdown]
# ## Reflection
#
# - What would a useful ablation reveal that a coefficient optimizer cannot?
# - Why should neighboring positions from one game stay in the same data partition?
# - Which design choices in a self-play match could bias a rating estimate?
#
# **Takeaway:** tune against labeled data, protect a true validation split, then evaluate decisions in controlled matches. Correctness and playing strength remain distinct, and neither a toy objective nor a few games is compelling evidence of strength.
#
# Next: [Chess formats](14_chess_formats.ipynb) examines position snapshots and game records.
