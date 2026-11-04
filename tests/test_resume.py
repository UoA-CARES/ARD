"""
Local tests for --resume. No docker, no GPU, no LLM calls, no HPC.

The LLM call (EurekaAgent.func_gen), the training backend (RewardEvaluator) and
the TensorBoard read (FitnessScorer.read_fitness) are replaced with small fakes.
Run from the repo root:  python -m pytest tests/test_resume.py -q
"""

import glob
import os
import sys
import types
import zlib
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# llm_agent imports openai at module load. Stub it when it is not installed,
# so these tests do not need network packages.
import importlib.util

if importlib.util.find_spec("openai") is None:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = mock.MagicMock
    sys.modules["openai"] = fake_openai

import main as ard_main  # noqa: E402
from src.reward_history import RewardHistory  # noqa: E402
from src.refinement.llm_agent import EurekaAgent  # noqa: E402
from src.evaluation.scorer import FitnessScorer  # noqa: E402


class CrashOnce(RuntimeError):
    """Raised by the fake evaluator to simulate a crash mid-run."""


class FakeEvaluator:
    """Stands in for RewardEvaluator. Writes a fake result file per candidate."""

    crash_tag = None          # tag to crash on (only once)
    evaluated = []            # every tag that was actually trained

    def __init__(self, output_dir, **_kwargs):
        self.output_dir = output_dir
        self.on_record_done = None

    def get_reward_template(self):
        return "TEMPLATE"

    def get_env_source(self):
        return "ENV"

    def evaluate(self, records, checkpoint_path=None):
        for record in records:
            if not record.has_method:
                record.status = "gen_failed"
                continue
            if record.status == "succeeded" and record.tb_path and os.path.exists(record.tb_path):
                continue  # already finished in a previous run
            if FakeEvaluator.crash_tag == record.tag:
                FakeEvaluator.crash_tag = None
                raise CrashOnce(f"simulated crash during {record.tag}")
            FakeEvaluator.evaluated.append(record.tag)
            work = os.path.join(self.output_dir, record.tag)
            os.makedirs(work, exist_ok=True)
            tb = os.path.join(work, "events.out.tfevents.fake")
            with open(tb, "w") as fh:
                fh.write("x")
            record.status = "succeeded"
            record.tb_path = tb
            record.checkpoint_path = None
            if self.on_record_done:
                self.on_record_done(record)
        return records


def fake_fitness(self, tb_file):
    """Deterministic fitness from the file path (no TensorBoard needed)."""
    return (zlib.crc32(tb_file.encode()) % 1000) / 1000.0


METHOD = "def compute_reward(self, *a):\n    return 0.0, {}\n"


@pytest.fixture
def patched(monkeypatch):
    FakeEvaluator.crash_tag = None
    FakeEvaluator.evaluated = []
    gen_calls = []

    def fake_func_gen(self, messages, seed=None):
        gen_calls.append(seed)
        return METHOD, f"```python\n{METHOD}```"

    monkeypatch.setattr(ard_main, "RewardEvaluator", FakeEvaluator)
    monkeypatch.setattr(EurekaAgent, "func_gen", fake_func_gen)
    monkeypatch.setattr(FitnessScorer, "read_fitness", fake_fitness)
    return gen_calls


SETTINGS_TMPL = {"tasks_repo": "unused", "runner": "local"}
TASK = {"task": "fake_task", "env_file": "env.py", "description": "d"}


def make_configs(tmp_path, iterations=2, base_seed=0):
    settings = dict(SETTINGS_TMPL, output_dir=str(tmp_path))
    refine = {
        "iteration": iterations, "num_eval": 1, "base_seed": base_seed,
        "agent": {"sample": 2, "model": "fake-model"},
    }
    return settings, refine


def find_run_dir(tmp_path):
    dirs = glob.glob(os.path.join(str(tmp_path), "fake_task", "*"))
    assert len(dirs) == 1, dirs
    return dirs[0]


# ---------------------------------------------------------------- unit tests

def test_history_round_trip_keeps_fields_and_null_fitness(tmp_path):
    history = RewardHistory(output_dir=str(tmp_path))
    good = history.new_record(iteration=1, index=0, phase="run", tag="iter1_run_0",
                              reward_method=METHOD, status="succeeded",
                              tb_path="/x/events", fitness=0.5)
    history.new_record(iteration=1, index=1, phase="run", tag="iter1_run_1",
                       reward_method=METHOD, status="build_failed",
                       eval_error="injection: boom")  # fitness stays -inf
    history.save_json()

    loaded = RewardHistory(output_dir=str(tmp_path))
    assert loaded.load_json() == 2
    recs = loaded.all()
    assert recs[0].tag == good.tag and recs[0].fitness == 0.5
    assert recs[0].tb_path == "/x/events"
    assert recs[1].fitness == float("-inf")
    assert recs[1].eval_error == "injection: boom"
    assert not os.path.exists(os.path.join(str(tmp_path), "reward_history.json.tmp"))


# ----------------------------------------------------------- end-to-end tests

def test_resume_after_crash_does_not_redo_finished_work(tmp_path, patched):
    gen_calls = patched
    settings, refine = make_configs(tmp_path)

    # Run 1: crash while training iter2_run_1. Iter1 and iter2 generation are done.
    FakeEvaluator.crash_tag = "iter2_run_1"
    with pytest.raises(CrashOnce):
        ard_main.run_refinement(settings, TASK, refine)
    run_dir = find_run_dir(tmp_path)
    first_run_gen = len(gen_calls)
    assert first_run_gen == 4  # 2 iterations x 2 samples, all generated before the crash
    trained_before_crash = list(FakeEvaluator.evaluated)
    assert "iter2_run_0" in trained_before_crash

    # Run 2: resume. Nothing is generated again, and finished candidates are not retrained.
    FakeEvaluator.evaluated = []
    history = ard_main.run_refinement(settings, TASK, refine, resume_dir=run_dir)

    assert len(gen_calls) == first_run_gen, "resume must not call the LLM again"
    assert "iter2_run_0" not in FakeEvaluator.evaluated, "finished candidate retrained"
    # Eval tags (e.g. iter1_run_0_eval_0) are new work after the loop, so only
    # exploration candidates count as "already done".
    assert not any(t.startswith("iter1_run_") for t in FakeEvaluator.evaluated)
    assert "iter2_run_1" in FakeEvaluator.evaluated, "crashed candidate must be trained"

    run_recs = [r for r in history.all() if r.phase == "run"]
    assert len(run_recs) == 4
    assert all(r.status == "succeeded" for r in run_recs)
    assert sum(r.selected_best for r in history.all() if r.phase == "run") == 1
    assert any(r.phase == "eval" for r in history.all())


def test_resume_refuses_changed_settings(tmp_path, patched):
    settings, refine = make_configs(tmp_path)
    ard_main.run_refinement(settings, TASK, refine)
    run_dir = find_run_dir(tmp_path)

    changed_seed = dict(refine, base_seed=7)
    with pytest.raises(ValueError, match="base_seed"):
        ard_main.run_refinement(settings, TASK, changed_seed, resume_dir=run_dir)


def test_resume_of_finished_run_does_nothing_new(tmp_path, patched):
    gen_calls = patched
    settings, refine = make_configs(tmp_path)
    ard_main.run_refinement(settings, TASK, refine)
    run_dir = find_run_dir(tmp_path)
    before = len(gen_calls)
    FakeEvaluator.evaluated = []

    ard_main.run_refinement(settings, TASK, refine, resume_dir=run_dir)
    assert len(gen_calls) == before
    assert FakeEvaluator.evaluated == []
