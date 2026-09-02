# Reward Genome Search

**ARD Stage 3 — procedure specification for the genetic-programming arm.**

A genetic-programming loop for LLM reward design. It replaces Eureka's single rolling
conversation with an archive, typed operators, and a fitness scale split into three bands.
This document specifies the algorithm, the experiment that tests it, and the code changes
it needs.

| | |
|---|---|
| Task | `Isaac-ARD-Repose-Cube-Shadow-Direct-v0` (Shadow Hand Repose) |
| Generations | 10 |
| Slots per generation | 10 |
| Eval seeds | 5 |
| Trainings per arm | 105 |

---

## 0. The rule that governs everything else

> The GP arm and the Eureka arm differ in **one thing only**: how the next batch of ten
> candidates is produced. Anything that is not that one thing must be identical in both
> arms, or removed from both.

Every design decision below was checked against this rule. It is why there are no retries
when a candidate crashes, why there is no calibration training run, and why the
tensor-shape manifest was dropped.

The goal is not the best possible reward search. The goal is a claim about genetic
programming that survives a reviewer asking "compared to what?"

---

## 1. Terms

| Term | Meaning |
|---|---|
| **Genome** | The source of one `_get_rewards` method. This is what mutates. |
| **Individual** | One genome, one training run, one fitness value. |
| **Generation** | One iteration of the loop. Always ten individuals. |
| **Slot** | One of the ten places in a generation. A slot is spent whether or not the candidate survives. |
| **Band** | Which region of the fitness scale a result lands in, split by T1 and T2. |
| **Archive** | Every individual ever evaluated, with its genome, error, band and diagnosis. |
| **Operator** | How a new genome is produced: clone, mutation, crossover, repair, or re-init. |
| **Arm** | One version of the experiment. There are two: Eureka and GP. |
| **Treatment** | The single difference between the arms, which is the thing being tested. |
| **Co-intervention** | Any *other* difference between the arms. A mistake: it makes the result impossible to attribute. |

---

## 2. Phase 0 — Thresholds

*Once per task, before any training. Costs zero training runs.*

One agent reads the task and writes three numbers. T1 separates "not doing the task" from
"doing something". T2 separates "works sometimes" from "works most of the time". T3 marks
saturation.

> T1 and T2 are **success rates translated into fitness units**, not fitness numbers
> guessed directly. T1 is the value a policy reaches when it is not doing the task. T2 is
> the value at which more than half of episodes succeed.

### 2.1 Why no calibration run is needed

On this task the observed noise floor spans roughly 0 to 1, while a genuinely working
policy scores around 30. That is a factor of thirty. When the gap is that wide, the
precision of T1 does not matter: any value between 2 and 5 separates the two populations
equally well. Spending training jobs to measure a floor to two decimal places, when a rough
analytic bound already sits thirty times away from the thing it separates, buys nothing and
costs budget.

### 2.2 The derivation, in order

| Step | Output | Source |
|---|---|---|
| A | What quantity is measured, and its kind: count, mean, moving average, rate, distance | Trace every symbol in `_log_fitness` back to environment state |
| B | What one step can contribute | Config bounds, tolerances, termination conditions |
| C | How a time series becomes one number | The scorer's aggregation, plus the mean over `num_envs` |
| D | **Chance level → T1** | What the metric reads when the policy never learns, derived from the termination logic |
| E | **Majority level → T2** | The value when more than half of episodes succeed |
| F | **Saturation → T3** | Analytic ceiling, minus a margin |
| G | Separation check | Assert `T1 < T2 < T3` with gaps wide relative to expected noise |

**Step D in practice.** Deriving the chance level is reading termination logic, not
guessing.

- For a **survival-time** metric it is a small positive number. Cartpole starts the pole
  inside ±0.25π and terminates past ±π/2, and with `sim.dt = 1/120`, `decimation = 2` and
  `episode_length_s = 5.0` the ceiling is 300 control steps, so undirected actions score a
  small fraction of that.
- For a **goal-hit** metric it is essentially zero. Random finger motion almost never lands
  a cube orientation inside tolerance, so `consecutive_successes` stays near 0, and one goal
  reach per episode on average puts T2 near 1.0.

### 2.3 Output, then frozen

Written once to `runs/<task>/task_analysis.json`:

```json
{
  "task": "Isaac-ARD-Repose-Cube-Shadow-Direct-v0",
  "measures": "smoothed count of goal orientations reached per episode",
  "type": "goal_count",
  "floor": 0.0,
  "ceiling": 50.0,
  "chance_level": 0.05,
  "chance_derivation": "random finger motion never lands inside tolerance",
  "success_unit": "one cube orientation matched within tolerance before a drop",
  "T1": 0.20,
  "T2": 1.00,
  "T3": 45.00,
  "separation_ok": true
}
```

> Once written, the thresholds **never change during a run**. If they move, the bands stop
> meaning anything and the results stop being reportable.

### 2.4 Handling seed noise without spending budget

Do not estimate seed variance before the run. Set T1 with a wide safety factor:

```
T1 = max(chance_upper_bound, 0.05 * ceiling)
```

so the boundary sits far outside the noise. Then measure the noise for free as the run
produces it, from the unmutated clone in each Mode C generation and from the five eval seeds
at the end. Record that variance and report it, but never let it move a threshold.

A result within the measured noise of T1 or T2 is tagged `borderline` and treated as the
lower band for parent selection.

### 2.5 When the agent designs the fitness function itself

The fitness function and its thresholds become one artifact, emitted in one call. Whoever
writes the metric already knows what it measures and what success means for it, so T1 and T2
stop being inferred and become specified.

This also gives a free quality gate: if the agent cannot produce a separable T1 and T2 for
its own metric, the metric cannot rank anything and is rejected before a single training run.

---

## 3. Phase 1 — Generation

*Every generation.*

You never ask how many children a parent should get. Every generation has exactly ten slots.
One number decides how they split: **the band of the best individual in the archive so far**.

### 3.1 Generation 1 — no parents exist yet

```
R R R R R R R R R R
```

All ten slots are re-init, no parent. Identical to Eureka's first iteration, so both arms
start from the same place.

### 3.2 Mode C — an elite exists (best is at or above T2)

```
C M M M M M M M M R
```

| Slots | Operator | Parent |
|---|---|---|
| 1 | clone, no change, new seed | the elite |
| 6 | mutation, careful | the elite |
| 2 | mutation, bolder | best band-B individual |
| 1 | re-init | none |

Seventy percent of the generation goes to the elite's lineage. The clone does two jobs at
once: it measures seed noise on a genome you already trained, and it guarantees the best
genome survives to the next generation unchanged (standard GP elitism). The three remaining
slots exist because a band-C reading is one noisy number from one seed, and a lucky seed
should not be able to take the whole population with it.

### 3.3 Mode B — best sits between T1 and T2

```
M M M M M M X X R R
```

| Slots | Operator | Parent |
|---|---|---|
| 3 | mutation | P1, the best band-B individual |
| 2 | mutation | P2 |
| 1 | mutation | P3 |
| 2 | crossover | P1 × P2, and P1 × P3 |
| 2 | re-init | none |

This is the exploration mode and it is where most of the search happens. Three parents
survive rather than one, because a single fitness reading is not enough evidence to commit
to one basin.

### 3.4 Mode A — everything is below T1

```
P P P P P R R R R R
```

| Slots | Operator | Parent |
|---|---|---|
| up to 5 | repair | one per band-A individual that passes the alignment test (§5.4) |
| rest | re-init | none |

### 3.5 How parents are ranked

**By rank, not by fitness value.** Sort eligible individuals by fitness, then hand out
children by position: first gets three, second gets two, third gets one.

Fitness-proportional selection (the classic roulette wheel) is wrong here. Fitness scales
differ per task and the numbers are noisy, so a candidate at 2.4 is not reliably fourteen
percent better than one at 2.1. Rank ignores the size of the gap and uses only the order,
which makes it scale-free and robust to exactly the seed noise this task has.

### 3.6 How crossover pairs are chosen

The best parent is always one half of the pair: **P1 × P2** and **P1 × P3**.

Crossover is semantic, not textual. The coder agent receives both parents' component lists
together with each component's observed statistics, and composes a child from them. Splicing
tensor code by AST breaks too often to be a useful operator.

### 3.7 What one mutation changes

Each child gets **exactly one** typed change, plus a written hypothesis and a predicted
metric effect. One change per child makes every generation an ablation: any fitness shift is
attributable to one identified edit, and the next generation's analysis can check whether the
prediction held.

| Type | Changes | Mode C | Mode B |
|---|---|---|---|
| `reweight` | a component's weight or scale | yes | yes |
| `retemp` | a temperature inside a transform | yes | yes |
| `retransform` | the functional form, e.g. linear to exponential | yes | yes |
| `add` | introduces a new component | no | yes |
| `remove` | deletes a component | no | yes |
| `rescope` | which state variable a component reads | no | yes |

Mode C allows only the first three. Those are the small, safe edits, and they are what
"careful and slight, with a good reason" means operationally. Mode B allows all six, because
you are still exploring and structural change is what you are exploring for.

### 3.8 Prompt assembly

Each child is one stateless call. There is no shared rolling conversation.

```
system   role, env source, hard requirements, output format

user     task description
         OPERATOR       which operator this child is, and its one allowed change type
         PARENT(S)      full source, training summary, band, diagnosis   [this child only]
         CONSTRAINTS    distinct craft-error signatures, with counts      [global, cap 10]
         ANTI-PATTERNS  distinct design failures, with diagnosis          [global, cap 8]
         ALREADY TRIED  genome signature + band + fitness, no code        [compact table]
```

### 3.9 Two safety rules

1. **Duplicate check.** Hash the structural signature before dispatch. A match against
   anything in the archive triggers regeneration. This costs an LLM call, not a training run,
   so it does not touch the budget.
2. **Unfillable slots become re-init.** If Mode B found only one band-B parent there is no
   crossover partner, so those slots fall back to re-init. The table always sums to ten.

---

## 4. Phase 2 — Training

*Every generation. Unchanged from the current pipeline.*

Ten candidates are injected, built, pushed and submitted to the CARES scheduler, and train
concurrently. Three things come back per job, all already copied into
`runs/<task>/<tag>/` by `HPCRunner.collect`:

| File | Contents | Used today |
|---|---|---|
| `logs/rl_games/.../summaries/events.*` | TensorBoard scalars | yes |
| `logs/container.log` | full stdout and stderr, including the Python traceback | **no** |
| `status.json` | `return_code`, `timed_out`, `runtime_seconds`, image, command | **no** |

> **Early stopping stays off in both arms.** Every run goes to its full `max_epochs`, so all
> runs have the same number of epochs and the final-window score is directly comparable
> between candidates.

If early stopping were on, a run's last epochs would by construction be the ones that failed
to beat the best, so a final-window mean would be biased downward by an amount that depends
on `patience`, and the window would cover a different number of epochs for each candidate.

> **A crashed candidate burns its slot.** No retry, no refill. Eureka behaves the same way,
> and refilling would change the sampling structure, which is the one thing the experiment
> cannot afford to change.

---

## 5. Phase 3 — Analysis

*Every individual, every generation.*

### 5.1 Gate 1 — did it run?

Read the traceback out of `container.log` and classify it. Most of this is a lookup, not an
LLM call. `status.json` adds two cheap signals: `timed_out` separates a hung job from a
crash, and a run that ends in under a minute did not train regardless of what its status says.

| Pattern | Class | Consequence |
|---|---|---|
| `SyntaxError`, `IndentationError`, `RewardInjectionError` | `injection` | caught before dispatch, regenerated at no training cost |
| `RuntimeError` shape or dtype, `AttributeError`, `TypeError`, `IndexError` | `craft` | signature joins the **constraint list** |
| NaN or inf in loss, `Expected parameter loc ...` | `design` | diagnosis joins the **anti-pattern list** |
| segfault, scheduler kill, timeout, empty output | `infra` | **not the genome's fault**, do not blame the candidate |

**What the existing 432 jobs actually contain.** 423 completed, 9 failed. Of the nine:
eight craft errors, one infra fault, zero design errors. The infra case is a segmentation
fault inside Isaac Sim startup (`XOpenDisplay` in the platform-info plugin), which has
nothing to do with the reward. That is why the `infra` class exists.

More importantly, **six of the eight craft failures are the same two mistakes**, repeated
across four separate runs over six days:

| Count | Pattern | Misconception |
|---|---|---|
| 4 | `self.fingertip_pos.view(self.num_envs, -1) - self.object_pos.unsqueeze(1)` | `fingertip_pos` is `(num_envs, 5, 3)` and `object_pos` is `(num_envs, 3)`; flattening one and unsqueezing the other does not align them |
| 2 | `torch.exp(-temp * torch.pi)` | `-temp * torch.pi` is a Python float, and `torch.exp` needs a tensor |

Two of those, `s50_iter1_run_3` and `s50_iter1_run_4`, are in the same batch. Nothing carried
the mistake forward, so the model kept making it. This is the empirical case for the
constraint list.

### 5.2 Gate 2 — training dynamics

Read from the scalar summary the pipeline already writes: initial ten percent, mid-point,
final ten percent, mean, standard deviation, maximum, minimum.

| Class | Signal | Meaning |
|---|---|---|
| `no_learning` | final ≈ initial for both fitness and reward | nothing was optimised |
| `converged` | reward rose, then the final segment is flat and stable | training finished |
| `still_improving` | fitness rising into the last window | budget-limited, not reward-limited |
| `collapsed` | rises then falls, or entropy collapses early | instability |

`still_improving` is kept, not discarded. A reward that was still working at the cutoff may
be good and simply under-trained, and throwing it away would discard the best genomes. It is
tagged `budget_limited` and reported as such.

### 5.3 Gate 3 — band

> Fitness is the **mean of the final ten percent of epochs**. The maximum over training is
> kept as a second field, `fitness_max`, for diagnosis only.

The current scorer ranks on the maximum over every epoch while the LLM is shown the
final-ten-percent number, so the model reflects on one value and is judged on another. Under
GP that breaks, because the analysis agent must assign a band from the same number it can
see. A large gap between `fitness` and `fitness_max` is itself the signal for `collapsed`.

**The decision table.** Dynamics and band are independent questions, so the outcome is a
grid, not a list.

| Dynamics | A (below T1) | B (T1 to T2) | C (above T2) |
|---|---|---|---|
| `no_learning` | cull | cull, lucky seed | verify before trusting |
| `collapsed` | cull | keep, tag `unstable` | keep, tag `unstable`, verify |
| `converged` | **deceptive** | **partial** | **elite** |
| `still_improving` | keep, tag `slow_start` | keep, tag `budget_limited` | elite, tag `budget_limited` |

### 5.4 Band A — the alignment test

A converged individual below T1 means the policy successfully maximised the reward and the
reward pointed somewhere else. This is reward hacking, and it is the most informative failure
the run produces.

Whether it can be repaired is decided **mechanically, without an LLM**. Take the set of
`self.*` attributes the fitness function reads. Take the set the candidate reward reads.
Compare:

| Intersection | Failure kind | Action |
|---|---|---|
| non-empty | **scaling**: the reward looks at the right quantity, something else drowned it out | `repair`, usually a one-line reweight |
| empty | **structural**: the reward never looks at the quantity being scored | `re-init`, nothing is worth keeping |

Default to re-init when the test is unclear. Showing a model a piece of code makes it produce
a variant of that code, so a deceptive parent tends to produce deceptive children. The
valuable part of a band-A individual is its diagnosis, not its source, and the anti-pattern
list carries the diagnosis into a re-init prompt without carrying the anchoring risk. Band A
also has no verified-good component to protect, unlike band B.

Repair adds one hard constraint to the prompt: **the new reward must contain at least one
term computed from the same environment state the fitness function reads**. That forces the
reward's support to overlap the fitness's support, which is the direct antidote to
misalignment.

---

## 6. Phase 4 — Archive

*Every generation.*

The archive is the only input to prompt assembly. Once that is true, a failed individual
stops being a special case: it is a record whose band is undefined and whose error fields are
populated. The current `if best is None` branch disappears.

**Why this also fixes a live bug.** Today the error prompt fires only when an *entire* batch
fails, and even then it fills its `{traceback_msg}` placeholder with a fixed sentence. The
real message sits in `eval_error`, is written to `reward_history.json`, and is never read
back. So that placeholder has never once contained a traceback. In the four existing runs the
branch never fired at all, since the worst iteration lost only three of ten candidates, which
means those runs remain a valid baseline after the fix.

### 6.1 New fields on each record

| Field | Purpose |
|---|---|
| `fitness` | primary score, now the final-ten-percent mean |
| `fitness_max` | maximum over epochs, diagnosis only |
| `stderr_tail` | traceback extracted from `container.log` |
| `error_class` | `none` / `injection` / `craft` / `design` / `infra` |
| `error_signature` | normalised key for dedup and counting, e.g. `RuntimeError:size-mismatch:fingertip_pos-object_pos` |
| `band` | `A` / `B` / `C` / `undefined` |
| `dynamics` | the four-way class from Gate 2 |
| `diagnosis` | the analysis agent's one-line finding |
| `genome_signature` | structural fingerprint: component names, quantities read, forms, weights, temperatures |
| `operator` | `reinit` / `mutate` / `crossover` / `repair` / `clone` |
| `parent_tags` | which individuals produced this one |
| `mutation_note` | the stated hypothesis and predicted effect |
| `preflight_attempts` | how many unparseable proposals were discarded before dispatch |

### 6.2 The two derived lists

- **Constraint list** — distinct `craft` error signatures. "You wrote the code wrong."
- **Anti-pattern list** — distinct `design` failures with their diagnosis. "The idea was wrong."

Both are global rather than per-lineage: a shape error is about the Isaac Lab API and a
hacking pattern is about the task, so every child should know about them. Both are
deduplicated by signature and carry a count, because "this error occurred four times" tells
the model it is a repeated mistake, and counting keeps the lists from growing without bound
across a hundred individuals.

### 6.3 What `feedback_text` becomes

Today it holds the message appended to a shared conversation. It becomes the full prompt this
specific child was generated from, so the exact input to any individual in the run can be
reconstructed afterwards. That is a much stronger position for a paper.

---

## 7. Phase 5 — Final evaluation

*Once, after the last generation. Unchanged.*

The best individual across all ten generations is re-trained on five seeds and reported as
mean and standard deviation over them. The mean is the reported score: a maximum over seeds
would report the luckiest seed, and would discard the variance the seeds were trained to
measure.

Total per arm, per task: **10 × 10 + 5 = 105 trainings**.

**T3 has no early-stop role**, since the budget is fixed. It stays useful as an operator
switch: an elite above T3 is near the ceiling, so further fine mutations all score the same
and teach nothing. At that point the same fixed slots are better spent on more unmutated
clones on new seeds. Given this task's seed variance, an elite verified across eight seeds is
a stronger reported result than eight indistinguishable mutations.

---

## 8. Experiment protocol

Two arms, one difference. The Eureka arm already exists: four runs (`s48`, `s49`, `s50`,
`s51`) on `Isaac-ARD-Repose-Cube-Shadow-Direct-v0`, each ten samples by ten iterations plus
five eval seeds, 432 jobs on disk under `~/hpc_outputs`.

> Run GP at **10 × 10** to match the existing baseline. The current `refineconfig.yaml` says
> 16 × 5, which would require re-running the baseline four times for no gain.

### GP arm only — this is the treatment

- Threshold agent, T1 / T2 / T3
- Band and dynamics classification
- The five operators and the slot-allocation modes
- The archive, with signatures and duplicate checking
- The constraint list and the anti-pattern list

### Both arms, identically

- Real traceback sent back to the model instead of a fixed sentence
- Fitness as the mean of the final ten percent of epochs
- Early stopping off
- Ten samples, ten iterations, five eval seeds
- Same task, same model, same temperature, same PPO configuration
- A crashed candidate burns its slot, with no retry

### Neither arm

- Crashed-slot retry
- Null calibration run
- Tensor shape manifest in the prompt

The traceback fix and the fitness statistic are **corrections, not improvements**. The Eureka
paper already feeds execution errors back; ARD's implementation simply failed to pass the
message through. Fixing it makes the baseline a more faithful copy of the published method,
not a handicapped one.

### Re-scoring the baseline costs nothing

Switching to the final-ten-percent mean requires no re-running. Every TensorBoard event file
for all 432 jobs is already on disk, so an offline script re-reads them and produces the
baseline's scores under the new statistic. Both arms then get scored identically, which is
what the comparison needs.

---

## 9. Build list

| Where | Change | Size |
|---|---|---|
| `evaluation/scorer.py:62` | Replace `max(e.value for e in events)` with the mean of the final ten percent. Store the maximum as `fitness_max`. | small |
| new, `evaluation/` | Traceback extractor. Read `<work_dir>/logs/container.log` from `Traceback (most recent call last):` to the exception line, plus the local runner's own capture. Map the reported file line back to the line of the generated method, since injection replaces the whole method and the offset is known. | small |
| new, `evaluation/` | Error classifier: pattern table to `error_class` plus `error_signature`. Read `status.json` for `timed_out` and `runtime_seconds`. | small |
| `reward_history.py` | Add the thirteen new record fields, all with defaults. Add `constraint_list()` and `antipattern_list()` views, and signature dedup. Nothing loads this file today, so the schema change is safe. | medium |
| `refinement/llm_agent.py:107` | Remove the rolling `self.messages` window. Generation becomes one stateless call per child, built from the operator, its parents, and the archive digest. This is the largest change. | large |
| new, `refinement/` | Analysis agent: dynamics, band, diagnosis, and the ranked list of proposed mutations with hypotheses. Threshold agent for Phase 0. | medium |
| new, `refinement/` | Alignment test: AST comparison of the `self.*` attributes read by the fitness function against those read by the candidate reward. | small |
| `main.py:137` | Replace the `if best is None` branch and the single-winner feedback with the mode-driven slot allocation. Per-individual analysis after every batch. | large |
| `configs/refineconfig.yaml` | `sample: 10`, `iteration: 10`, `num_eval: 5`. | trivial |
| offline script | Re-score the 432 existing baseline jobs under the new fitness statistic. | small |

**Out of scope but worth considering.** `reward_history.json` is currently write-only, with
no loader. A hundred trainings on the cluster is many hours, and if a run dies at generation
eight the whole fixed budget is lost. Once the archive is the single source of truth, a loader
is roughly twenty lines and makes a run restartable.

---

## 10. Known limitations

### GP also has a longer memory, and that is a second difference

Eureka's conversation remembers only the last iteration. The GP archive remembers all ten.
This was not added on purpose, it comes for free with the archive, and it cannot be removed
without breaking GP.

So if GP wins there are two possible causes: the operators are better, or GP simply remembers
more. This experiment cannot separate them. State it in the paper rather than hide it, and
list the separating experiment as future work: run GP again with the archive memory cut to one
generation, and see how much of the win survives.

### Single task, single fitness function

The baseline is four runs on one task. Any claim is about Shadow Hand Repose under
`consecutive_successes`, not about reward design in general.

### Thresholds are derived, not measured

T1 and T2 come from reading code, not from calibration runs. This is defensible here because
the noise-to-signal gap is roughly thirty times, so threshold precision does not affect band
assignment. On a task where that gap is small, the derivation would need the measurement, and
the separation check in step G is what would catch it.

---

## Related documents

- [`README.md`](README.md) — what ARD is and how to run it.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the Stage 2 pipeline this loop is built on: reward
  injection, the local and HPC backends, and the fitness isolation guarantee in the task layer.
