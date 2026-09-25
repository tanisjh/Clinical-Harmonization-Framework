# AI Coding Agent Rules — NIH Biowulf

This workspace is on the NIH Biowulf shared HPC cluster.

These rules apply to any AI coding agent operating in this workspace.

The **workspace** is the directory containing this `AGENTS.md` file and its descendants.

## 1. Core operating principles

- Treat Biowulf as a shared HPC system, not a standalone workstation.
- Prefer read-only inspection before making changes.
- Stay within the scope of the user's current request.
- Do not silently broaden a coding task into a larger refactor, repair, experiment, or cleanup.
- If completing the requested task appears to require materially broader work, stop and explain what additional work would be required.
- Never claim that a command, test, experiment, validation, or job was run unless it was actually run.
- Preserve existing user work, including uncommitted and untracked files.
- Inspect repository state before making substantial changes.

## 2. Workspace boundaries

### Writes

- Modify files only inside this workspace unless the user explicitly authorizes changes elsewhere.
- Do not delete, overwrite, move, or modify files outside this workspace without explicit approval.
- Do not modify external repositories, dependency checkouts, shared datasets, shared software installations, or other users' files.

### Reads

Targeted read-only inspection outside the workspace is allowed when it is genuinely needed to complete the task, for example:

- inspecting a pinned dependency checkout;
- reading canonical/reference source code or documentation;
- inspecting an installed tool or library;
- checking a shared dataset's metadata;
- verifying an external repository's implementation.

Do not perform broad recursive searches of unrelated directories or inspect unrelated user data.

When correctness depends on an external or pinned dependency, prefer inspecting the actual pinned source/version over trusting inherited notes, prior reports, or assumptions.

## 3. Scratch space

- Never deliberately use shared `/tmp` for scratch work.
- Use workspace-local scratch space:

  `./.scratch/`

- Create additional subdirectories beneath `./.scratch/` as needed.
- Preserve existing scratch material unless the task specifically requires replacing a named file.
- Clean temporary scratch created for a task when it is no longer needed, unless it is intentionally retained for reproducibility, debugging, or handoff.

## 4. Risk-based approval policy

The purpose of approval prompts is to identify actions with meaningful risk.

Do **not** ask for approval for ordinary, low-risk coding work inside the workspace.

### Normally allowed without additional approval

Examples include:

- read-only file inspection;
- targeted read-only inspection outside the workspace as described above;
- editing files inside the workspace;
- creating workspace-local scratch files;
- running ordinary unit tests, linters, formatters, and small deterministic scripts;
- inspecting Git state with commands such as `git status`, `git diff`, `git log`, and `git show`;
- checking read-only Slurm job status or scheduler information;
- creating a local branch when the user's task calls for one;
- creating a local commit when the user's prompt explicitly requests a commit.

### Require explicit user approval

Ask before performing actions with meaningful risk, including:

- submitting a Slurm job;
- cancelling, requeuing, holding, releasing, modifying, or otherwise changing an existing Slurm job;
- launching large, recursive, highly parallel, long-running, or resource-intensive computations;
- modifying anything outside this workspace;
- deleting substantial amounts of data;
- recursive destructive filesystem operations;
- installing, upgrading, downgrading, or removing packages or environments when this could affect more than the current workspace;
- changing permissions, ownership, shell configuration, system configuration, or shared software;
- killing or modifying running processes not created as part of the current task;
- uploading or transmitting workspace data to an external service;
- pushing Git commits or branches to a remote;
- merging branches;
- rebasing;
- rewriting Git history;
- deleting branches;
- force operations;
- commands that could discard uncommitted work;
- destructive Git operations such as `git reset --hard`, `git clean -fd`, or equivalent;
- any action whose consequences are uncertain and potentially difficult to reverse.

If approval is required, explain briefly what action is proposed and why it is potentially consequential.

Do not ask for approval merely because a harmless command changes a workspace-local file.

## 5. Biowulf and Slurm rules

- Never run agentic workloads on Biowulf login nodes.
- Agentic coding sessions and agentic commands must run inside an appropriate Slurm compute job.
- Do not submit Slurm jobs unless explicitly requested.
- Do not modify or cancel existing Slurm jobs unless explicitly requested.
- Read-only scheduler inspection is allowed.
- Do not run large or resource-intensive workloads without explicit approval.
- Respect requested CPU, memory, GPU, and wall-time allocations.
- Do not deliberately exceed Slurm resource allocations.
- Treat filesystem, scheduler, network, and compute resources as shared resources.

Biowulf does not currently provide the normal OS sandboxing assumptions used by some agentic coding tools. Do not assume sandbox restrictions provide a security boundary.

### Python environment

- Respect the Python environment defined for the current project.
- Before running Python commands, determine which environment the project expects to use. This may be a project-local virtual environment, a conda environment, a Biowulf module-provided Python, or another explicitly documented environment.
- Prefer the Python interpreter associated with that project environment rather than a system-default interpreter.
- When a project-local virtual environment exists, use its interpreter explicitly when practical, for example `./.venv/bin/python`.
- When a conda environment is required, ensure the intended conda environment is active before running Python commands.
- When a Biowulf Python module or other module is required, ensure the required module is loaded before using the environment.
- Do not assume a particular Python version unless it is specified by the project configuration, documentation, environment, or user.
- Do not silently fall back to `/usr/local/bin/python`, `/usr/bin/python`, another system Python, or a different environment if the intended project interpreter is unavailable.
- For package operations, use the package manager associated with the intended project environment, preferably through that environment's Python when applicable, such as `<project-python> -m pip`.
- Do not create, recreate, delete, replace, upgrade, or otherwise modify a project's Python or conda environment unless the user's task explicitly requires it or the user approves it.
- If the expected environment cannot be identified or activated reliably, stop and report the problem rather than proceeding with a different Python environment.

## 6. Secrets and credentials

Never expose, print, copy, transmit, commit, or otherwise reveal:

- passwords;
- API keys;
- access tokens;
- authentication cookies;
- SSH private keys;
- credential files;
- secrets stored in environment variables;
- other authentication material.

Do not add secrets to logs, handoffs, test fixtures, Git commits, prompts, or external services.

If a command may print sensitive material, inspect or constrain it appropriately before running it.

## 7. Git discipline

Before substantial coding work:

- inspect `git status`;
- identify the current branch and HEAD;
- note any pre-existing modified or untracked files;
- do not overwrite unrelated user changes.

During work:

- keep changes narrowly scoped to the requested task;
- inspect diffs before committing;
- do not silently modify unrelated files.

Commits:

- If the user's prompt explicitly requests a commit, create one narrowly scoped local commit for the approved task.
- Otherwise, do not create a commit unless requested.
- Do not push unless explicitly requested.
- Do not rewrite existing history unless explicitly requested.
- Do not add AI attribution trailers or metadata to commit messages, including:
  - `Co-authored-by`
  - `Claude-Session`
  - similar AI attribution lines.

## 8. Coding and verification discipline

- Prefer the smallest correct change over broad refactoring.
- Preserve existing interfaces and behavior unless the task requires changing them.
- When fixing a defect, reproduce or characterize the old behavior when practical.
- Add focused regression tests when appropriate.
- Run relevant existing tests after changes.
- Distinguish:
  - tests actually run;
  - tests not run;
  - assumptions not directly verified.
- Do not weaken or delete a failing test merely to make a suite pass unless the test itself is demonstrably incorrect and the change is part of the requested task.
- Do not manufacture evidence to match an expected result.
- Report negative findings when an expected defect is not actually present.
- Re-derive important external contracts from pinned source or authoritative documentation when practical.

## 9. Coding-task scope

A **coding task** is the body of work performed in response to one user prompt requesting implementation, debugging, testing, refactoring, or another code change.

For each coding task:

- record the intended scope before substantial modification;
- remain within that scope;
- document material deviations;
- do not begin the next coding task without a new user request.

If a task uncovers additional work that should be done later, record it as a follow-up rather than silently implementing it.

## 10. Session handoff protocol

Maintain a lightweight recovery handoff during every coding-agent session.

The purpose is to preserve operational state if the network, terminal, agent session, or model context is lost.

### Handoff location

At the beginning of each new coding-agent session, create a new file under:

`./.scratch/handoffs/`

Use a date-and-time-based filename such as:

`YYYY-MM-DD-HHMM.md`

Use local session time.

Do not overwrite an earlier session's handoff.

### At session start

Before substantial work:

1. inspect the current Git state;
2. inspect the most recent relevant handoff if one exists;
3. create the new session handoff;
4. record:
   - repository/workspace path;
   - current branch;
   - current HEAD;
   - working-tree status;
   - relevant untracked or modified files;
   - current known objective;
   - important prior decisions needed to continue safely.

### At the start of each coding task

Update the current session handoff with:

- the user's requested task;
- intended scope;
- expected starting checkpoint;
- any important constraints or decisions.

Keep this concise.

### After each coding task

Update the handoff with:

- what was changed;
- important technical decisions;
- files materially changed;
- tests and validations actually run;
- exact test results when important;
- commit hash if a commit was created;
- current branch and HEAD;
- current working-tree state;
- unresolved issues or risks;
- recommended next step.

### During a long coding task

Add an intermediate checkpoint only at meaningful boundaries, for example:

- investigation complete;
- implementation complete;
- before a risky operation;
- before a long validation run;
- before committing.

Do not update the handoff after every command or tool call.

### Handoff content rules

The handoff should be a concise recovery checkpoint, not a transcript.

Do not include:

- hidden chain-of-thought;
- full conversation transcripts;
- tool-call logs;
- large command outputs;
- secrets or credentials;
- irrelevant exploratory detail.

Prefer current state, decisions, evidence, unresolved issues, and next actions.

## 11. Session interruption and recovery

If a session appears to be ending, losing connectivity, or running out of usable context:

- update the current handoff before doing additional nonessential work;
- record the exact current Git state;
- record whether implementation is complete, partially complete, tested, committed, or uncommitted;
- record the safest next action.

A new agent session should use the latest handoff as orientation, but it must still verify current repository state rather than blindly trusting the handoff.

## 12. Safety takes precedence

When instructions conflict:

1. protect credentials and user data;
2. avoid destructive or difficult-to-reverse actions;
3. protect shared Biowulf resources;
4. preserve existing user work;
5. stay within the user's requested scope.

If an action is genuinely ambiguous and potentially dangerous, ask before proceeding.

For routine, workspace-local, low-risk coding actions, proceed without unnecessary approval prompts.
