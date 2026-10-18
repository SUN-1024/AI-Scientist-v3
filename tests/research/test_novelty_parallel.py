"""Independent novelty calls overlap; cancellation joins all owned work."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import CancelledError

import pytest

from src import review_backend
from src.research.novelty_review import InvalidReviewResume, review_novelty
from tests.paper import test_review_process_lifecycle as lifecycle_fixtures
from tests.research import test_novelty_gate as fixtures

study = fixtures.study
tree_factory = lifecycle_fixtures.tree_factory


def _complete(prompt, work_dir, review):
    work_dir.mkdir()
    native = str(uuid.uuid4())
    raw = json.dumps({"protocolVersion": 1, "type": "result", "exitCode": 0,
                      "data": {"finalMessage": json.dumps(review), "nativeSessionId": native}})
    (work_dir / "stdout.json").write_text(raw)
    transport = {"status": "completed", "independent_context": True,
                 "native_session_id": native, "observed_provider": "fixture",
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "stdout_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    (work_dir / "receipt.json").write_text(json.dumps(transport))
    return review, transport


@pytest.mark.parametrize("reviewer_count", [2, 4])
def test_agent_reviewers_overlap_and_reverse_completion_keeps_stable_order(
    study, tmp_path, monkeypatch, reviewer_count,
):
    task, cfg, agenda, review = study
    barrier = threading.Barrier(reviewer_count, timeout=3)
    completed = [threading.Event() for _ in range(reviewer_count)]
    order, cancellations = [], []

    def run(prompt, *, work_dir, cancel_event, **settings):
        index = int(work_dir.parent.name.rsplit("_", 1)[1]) - 1
        cancellations.append(cancel_event)
        barrier.wait()  # A serial implementation cannot pass this barrier.
        if index + 1 < reviewer_count:
            assert completed[index + 1].wait(3)
        result = _complete(prompt, work_dir, {**review, "rationale": f"Reviewer {index + 1}."})
        order.append(index + 1)
        completed[index].set()
        return result

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    output = tmp_path / "reviews"
    report = review_novelty(task, agenda, cfg, output, reviewer_count=reviewer_count,
                            reviewer_settings={"agent": "codex", "model": "fixture"})
    assert order == list(range(reviewer_count, 0, -1))
    assert len({id(event) for event in cancellations}) == 1
    assert report["status"] == "supported"
    assert [row["assessment"]["rationale"] for row in report["model_judgments"]] == [
        f"Reviewer {index}." for index in range(1, reviewer_count + 1)
    ]


def test_http_reviewers_also_overlap(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    barrier = threading.Barrier(2, timeout=3)
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")

    def request(*_args, **_kwargs):
        barrier.wait()
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review)}}]}

    monkeypatch.setattr("src.research.model_search._http_json", request)
    assert review_novelty(task, agenda, cfg, tmp_path / "reviews")["status"] == "supported"


def test_failure_preserves_other_success_and_resume_does_not_reissue_cache(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    barrier = threading.Barrier(2, timeout=3)
    calls = []

    def run(prompt, *, work_dir, **settings):
        name = work_dir.parent.name
        calls.append(name)
        barrier.wait()
        if name == "reviewer_1":
            work_dir.mkdir()
            (work_dir / "stdout.json").write_text("partial failed response")
            raise review_backend.AgentReviewError("fixture provider failure")
        return _complete(prompt, work_dir, review)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture"}
    report = review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    assert report["status"] == "needs_evidence" and len(report["model_judgments"]) == 1
    success = (output / "reviewer_2/receipt.json").read_bytes()
    failed = (output / "reviewer_1/receipt.json").read_bytes()

    def retry(prompt, *, work_dir, **settings):
        calls.append(work_dir.parent.name)
        assert work_dir.parent.name == "reviewer_1"
        return _complete(prompt, work_dir, review)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", retry)
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"
    assert calls.count("reviewer_1") == 2 and calls.count("reviewer_2") == 1
    assert (output / "reviewer_2/receipt.json").read_bytes() == success
    assert next((output / "retry_attempts/reviewer_1").glob("*/receipt.json")).read_bytes() == failed
    assert next((output / "retry_attempts/reviewer_1").glob("*/transport/stdout.json")).read_text() == "partial failed response"
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"
    assert len(calls) == 3
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == before


def test_all_retry_archives_are_preflighted_before_any_retry(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    calls = []

    def fail(prompt, *, work_dir, **settings):
        calls.append(work_dir.parent.name)
        raise review_backend.AgentReviewError("fixture provider failure")

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", fail)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture"}
    review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    outside = tmp_path / "outside"
    outside.mkdir()
    archive = output / "retry_attempts"
    archive.mkdir()
    (archive / "reviewer_2").symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidReviewResume, match="archive"):
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert len(calls) == 2
    assert not (archive / "reviewer_1").exists()
    assert not list(outside.iterdir())


@pytest.mark.parametrize("interruption", ["keyboard", "sigterm", "unexpected"])
def test_main_interruption_waits_for_every_reviewer_cleanup_and_can_resume(
    study, tmp_path, monkeypatch, interruption,
):
    task, cfg, agenda, review = study
    barrier = threading.Barrier(2, timeout=3)
    cleaned = set()
    cancellation_seen = []
    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def run(prompt, *, work_dir, cancel_event, **settings):
        name = work_dir.parent.name
        try:
            barrier.wait()
            if name == "reviewer_1":
                if interruption == "keyboard":
                    raise KeyboardInterrupt
                if interruption == "unexpected":
                    raise RuntimeError("unexpected reviewer failure")
                os.kill(os.getpid(), signal.SIGTERM)
            assert cancel_event.wait(3)
            cancellation_seen.append(name)
            # A second SIGTERM must not interrupt the owner's pending join.
            if interruption == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.03)
            raise review_backend.AgentReviewError("independent reviewer cancelled")
        finally:
            cleaned.add(name)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture"}
    expected = {"keyboard": KeyboardInterrupt, "sigterm": SystemExit, "unexpected": RuntimeError}[interruption]
    with pytest.raises(expected) as error:
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    if interruption == "sigterm":
        assert error.value.code == 143
    assert cleaned == {"reviewer_1", "reviewer_2"}
    assert "reviewer_2" in cancellation_seen
    assert {sig: signal.getsignal(sig) for sig in original_handlers} == original_handlers
    assert not (output / "novelty_report.json").exists()
    assert all(json.loads(path.read_text())["status"] == "failed" for path in output.glob("reviewer_*/receipt.json"))
    calls = []

    def retry(prompt, *, work_dir, **settings):
        calls.append(work_dir.parent.name)
        return _complete(prompt, work_dir, review)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", retry)
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"
    assert sorted(calls) == ["reviewer_1", "reviewer_2"]
    assert len(list((output / "retry_attempts").glob("reviewer_*/*/receipt.json"))) == 2


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_during_executor_thread_registration_defers_raise_until_cleanup(
    study, tmp_path, monkeypatch, signum,
):
    task, cfg, agenda, review = study
    entered = threading.Event()
    cleaned = threading.Event()
    registration_returned = threading.Event()
    calls = []
    original_start = threading.Thread.start
    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def run(prompt, *, work_dir, cancel_event, **settings):
        calls.append(work_dir.parent.name)
        entered.set()
        try:
            assert cancel_event.wait(3)
            # Repeated signals during shutdown must not interrupt the join.
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.03)
            raise review_backend.AgentReviewError("independent reviewer cancelled")
        finally:
            cleaned.set()

    def start(thread):
        original_start(thread)
        if thread.name.startswith("novelty-review"):
            assert entered.wait(3)
            # ThreadPoolExecutor has started this thread but has not yet added
            # it to the set joined by shutdown(). The handler must not raise.
            os.kill(os.getpid(), signum)
            registration_returned.set()

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    monkeypatch.setattr(threading.Thread, "start", start)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture"}
    with pytest.raises(SystemExit if signum == signal.SIGTERM else KeyboardInterrupt) as error:
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    if signum == signal.SIGTERM:
        assert error.value.code == 143
    assert registration_returned.is_set() and cleaned.is_set()
    assert calls == ["reviewer_1"]
    assert not (output / "reviewer_2").exists()
    assert {sig: signal.getsignal(sig) for sig in original_handlers} == original_handlers
    monkeypatch.setattr(threading.Thread, "start", original_start)
    monkeypatch.setattr(review_backend, "run_fresh_agent_json",
                        lambda prompt, *, work_dir, **settings: _complete(prompt, work_dir, review))
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"


@pytest.mark.parametrize("first_cause", ["signal", "exception"])
def test_signal_handler_never_reenters_cancellation_event_lock(study, tmp_path, monkeypatch, first_cause):
    task, cfg, agenda, _review = study
    barrier = threading.Barrier(2, timeout=3)
    cleaned = set()

    class SignalDuringSet(threading.Event):
        setting = False
        injected = False

        def set(self):
            # Fail fast if a future handler regresses to Event.set. Without this
            # test guard that handler would deadlock on the real Condition lock.
            assert not self.setting, "signal handler reentered Event.set"
            self.setting = True
            try:
                with self._cond:
                    if not self.injected:
                        self.injected = True
                        os.kill(os.getpid(), signal.SIGTERM)
                        os.kill(os.getpid(), signal.SIGINT)
                    self._flag = True
                    self._cond.notify_all()
            finally:
                self.setting = False

    cancellation = SignalDuringSet()

    def run(prompt, *, work_dir, cancel_event, **settings):
        name = work_dir.parent.name
        try:
            barrier.wait()
            if name == "reviewer_1":
                if first_cause == "exception":
                    raise RuntimeError("trigger cancellation from ordinary control flow")
                os.kill(os.getpid(), signal.SIGINT)
            assert cancel_event.wait(3)
            raise review_backend.AgentReviewError("independent reviewer cancelled")
        finally:
            cleaned.add(name)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    expected = KeyboardInterrupt if first_cause == "signal" else SystemExit
    with pytest.raises(expected) as error:
        review_novelty(task, agenda, cfg, tmp_path / "reviews",
                        reviewer_settings={"agent": "codex", "model": "fixture"},
                        cancel_event=cancellation)
    if first_cause == "exception":
        assert error.value.code == 143
    assert cancellation.injected and cancellation.is_set()
    assert cleaned == {"reviewer_1", "reviewer_2"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-tree signals")
def test_external_cancel_from_non_main_thread_joins_real_process_trees_and_can_resume(
    study, tmp_path, monkeypatch, tree_factory,
):
    task, cfg, agenda, review = study
    trees = {f"reviewer_{index}": tree_factory(f"reviewer-{index}") for index in (1, 2)}
    original = review_backend.run_fresh_agent_json
    cancellation = threading.Event()
    cleaned = set()

    def run(prompt, *, work_dir, cancel_event, **settings):
        name = work_dir.parent.name
        assert cancel_event is cancellation

        def exited(process):
            assert process.poll() is not None
            assert not trees[name].live_pids()
            cleaned.add(name)

        return original(prompt, work_dir=work_dir, cancel_event=cancel_event,
                        environment=trees[name].environment, on_process_exit=exited, **settings)

    monkeypatch.setattr(review_backend, "run_fresh_agent_json", run)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture", "timeout_seconds": 20}
    outcome = {}
    original_signal = signal.signal

    def only_main_thread(*args):
        assert threading.current_thread() is threading.main_thread()
        return original_signal(*args)

    monkeypatch.setattr(signal, "signal", only_main_thread)

    def call():
        try:
            outcome["result"] = review_novelty(task, agenda, cfg, output, reviewer_settings=settings,
                                               cancel_event=cancellation)
        except BaseException as error:
            outcome["error"] = error

    owner = threading.Thread(target=call)
    owner.start()
    try:
        for tree in trees.values():
            tree.wait_ready()
        cancellation.set()
        owner.join(8)
        assert not owner.is_alive()
        assert isinstance(outcome.get("error"), CancelledError)
        assert cleaned == {"reviewer_1", "reviewer_2"}
        assert all(not tree.live_pids() for tree in trees.values())
        assert not (output / "novelty_report.json").exists()
        for receipt in output.glob("reviewer_*/transport/receipt.json"):
            assert json.loads(receipt.read_text())["status"] == "failed"
    finally:
        cancellation.set()
        owner.join(8)
    monkeypatch.setattr(review_backend, "run_fresh_agent_json",
                        lambda prompt, *, work_dir, **settings: _complete(prompt, work_dir, review))
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)["status"] == "supported"
    assert len(list((output / "retry_attempts").glob("reviewer_*/*/transport/receipt.json"))) == 2
