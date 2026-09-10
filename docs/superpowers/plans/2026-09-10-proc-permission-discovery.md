# Proc Permission Discovery Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow AgentOPSD preflight to inspect a mixed-ownership `/proc` table without failing on unrelated protected processes, while retaining fail-closed checks for possible AgentOPSD processes.

**Architecture:** Split process discovery from strict process revalidation. Discovery filters foreign UIDs before protected reads and represents inaccessible optional metadata explicitly; target selection rejects incomplete AgentOPSD candidates, while termination continues to use strict `read_process` identity checks.

**Tech Stack:** Python 3, Linux procfs, pytest

## Global Constraints

- Do not launch training or any resource-consuming workload during this fix.
- Do not weaken strict process identity validation before sending signals.
- Keep changes limited to the cleanup module and its focused unit tests.

---

### Task 1: Make procfs discovery permission-aware

**Files:**
- Modify: `examples/process_cleanup/agentopsd_process_cleanup.py`
- Test: `tests/test_agentopsd_process_cleanup.py`

**Interfaces:**
- Consumes: Linux `/proc/<pid>/{status,stat,cmdline,cwd,environ}` entries.
- Produces: `read_process(pid, proc_root, expected_uid=None, allow_incomplete=False)` and `ProcessInfo.inspection_complete` for safe discovery and strict revalidation.

- [x] **Step 1: Write failing discovery tests**

Add tests proving discovery skips a foreign UID before protected reads, retains unrelated same-UID processes with incomplete optional metadata, and rejects an incomplete AgentOPSD driver candidate.

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_agentopsd_process_cleanup.py`

Expected: new tests fail because UID-aware partial discovery is not implemented.

- [x] **Step 3: Implement the minimal discovery behavior**

Add an `inspection_complete` flag with a backward-compatible default. Let `iter_processes` request current-UID, incomplete-tolerant reads; skip foreign UIDs before `cwd` and `environ`; mark protected optional metadata incomplete. Fail selection when such an incomplete process identifies itself as the AgentOPSD driver or a legacy worker.

- [x] **Step 4: Run focused and regression tests**

Run: `pytest -q tests/test_agentopsd_process_cleanup.py tests/test_agentopsd_launcher_cleanup.py`

Expected: all tests pass.

- [x] **Step 5: Reproduce the original environment safely**

Run a read-only Python snippet that enumerates `iter_processes()` and confirms PID 1 no longer aborts enumeration. Do not invoke `preflight`, because it can terminate stale processes.

Expected: enumeration completes and contains only the current UID.
