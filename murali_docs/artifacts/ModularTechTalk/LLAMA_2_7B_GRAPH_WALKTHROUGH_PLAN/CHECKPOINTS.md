# Git checkpoints

| Version | Git reference | What it preserves |
|---|---|---|
| Original guide | `llama-calltrace-baseline-20261101` / `e89259a8d3` | Complete guide before the first prototype |
| Six-tree replacement | `llama-text-callstack-20261101` / `d746b17324` | Separate reader for QnA 1.1.1–1.1.6, collapsible startup map and verified source links |
| Combined startup tree | `llama-unified-callstack-20261101` | One continuous call tree, with descriptive MAX Serve / Llama 2 7B startup filenames |

On 2026-11-01, the rejected prototype was removed from this task folder before
the replacement was built. The tracked folder was verified identical to the
baseline with `git diff --exit-code llama-calltrace-baseline-20261101 -- <task-folder>`.
The rejected, uncommitted draft is preserved separately as Git stash
`73513071e3b0f0e643146a15304feeba7a0fffdd`; it is not in the replacement.

Implementation source remains pinned to
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. These checkpoints change only
the documentation task folder; they do not change MAX implementation code.

## Inspect the checkpoints

From the repository root:

```bash
git show --stat llama-calltrace-baseline-20261101
git show --stat llama-text-callstack-20261101
git show --stat llama-unified-callstack-20261101
git diff llama-text-callstack-20261101 llama-unified-callstack-20261101 -- murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/
```

## Roll back while preserving history

First inspect `git status` and preserve any later edits to the task folder.
To undo only the merge/rename and return to the six-tree reader:

```bash
git revert -s llama-unified-callstack-20261101
```

To then return to the original guide baseline, also run
`git revert -s llama-text-callstack-20261101`.
Each command creates a new rollback commit. Later edits to the same files can
require conflict resolution. No hard reset or repository-wide clean is needed.
