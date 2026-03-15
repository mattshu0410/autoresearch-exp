# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

### Evolutionary protocol (gene pool)

The experiment loop is driven by a **gene pool** (`evo/gene_pool.json`) containing architecture papers and winning code snapshots. Each entry has a fitness weight that evolves based on experimental results. The crossover engine is in `evo/crossover.py`.

**SETUP (once, before the loop starts):**

1. Read `evo/gene_pool.json` to see what's in the pool.
2. Run the baseline `train.py` as-is. Record val_bpb.
3. Update the baseline entry's `val_bpb` in `evo/gene_pool.json`.

**LOOP FOREVER:**

1. **Ask the scheduler** what to do this iteration:
   ```
   python -c "
   from evo.crossover import load_pool, should_crossover
   pool = load_pool()
   d = should_crossover(pool)
   print(f'Action: {d[\"action\"]}')
   print(f'Reason: {d[\"reason\"]}')
   print(f'Experiment: {d[\"experiment\"]}, xover_prob: {d[\"base_prob\"]:.3f}, consec_losses: {d[\"consecutive_losses\"]}')
   "
   ```
   The schedule is configured in `evo/gene_pool.json` under `"schedule"` (type, xover_start, xover_end, n_total, staleness_trigger). It anneals crossover probability over time — high early (explore architectures), low late (tune hyperparameters).

   - **If `action == "tune"`**: Edit `train.py` directly — tweak hyperparameters, optimizer settings, small architectural changes. Use your own judgment based on recent results. Skip to step 3 (git commit).
   - **If `action == "crossover"`**: Continue to step 2.

2. **Sample parents and crossover**: Run `python -c "from evo.crossover import load_pool, sample_parents, get_parent_info; pool = load_pool(); a, b = sample_parents(pool); print(f'Parent A: {a}'); print(f'Parent B: {b}'); ia = get_parent_info(pool, a); ib = get_parent_info(pool, b); print(f'A type={ia[\"type\"]}, summary={ia[\"summary\"][:100]}'); print(f'B type={ib[\"type\"]}, summary={ib[\"summary\"][:100]}'); print(f'A content: {ia[\"content_path\"]}'); print(f'B content: {ib[\"content_path\"]}')"` to pick two parents weighted by fitness. Then spawn a subagent (Task tool) to do the blending. The subagent should:
   - Read both parent content files (the full paper markdown or code snapshot)
   - Read the current `train.py`
   - Write a new `train.py` that coherently blends ideas from both parents
   - Update the MANIFEST comment block at the top of `train.py`

   **IMPORTANT**: Use a subagent for this step to avoid loading large paper files into your main context. The subagent prompt should include:
   - The content paths for both parents
   - What type each parent is (paper or code) and their summaries
   - The path to the current `train.py`
   - Instructions to blend ideas and write a valid, runnable `train.py`
   - All constraints from this file (single file, imports from prepare.py, etc.)

3. **git commit** the new `train.py`.

4. **Run the experiment**: `uv run train.py > run.log 2>&1`

5. **Read results**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
   - If grep is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace. If fixable, fix and re-run. Otherwise, skip.

6. **Update gene pool weights**: Run:
   ```
   python -c "
   from evo.crossover import load_pool, save_pool, update_weights
   pool = load_pool()
   pool = update_weights(pool, ('PARENT_A_ID', 'PARENT_B_ID'), won=BOOL, delta_bpb=DELTA, best_bpb=BEST)
   save_pool(pool)
   "
   ```
   Where `delta_bpb = best_bpb - new_bpb` (positive = improvement), and `won` is True if val_bpb improved.

7. **If val_bpb improved** (lower):
   - Keep the git commit (advance the branch).
   - Register the winner:
     ```
     python -c "
     from evo.crossover import load_pool, save_pool, register_winner
     pool = load_pool()
     train_py = open('train.py').read()
     pool = register_winner(pool, ('PARENT_A_ID', 'PARENT_B_ID'), train_py, NEW_BPB, BEST_BPB, 'description of what changed')
     save_pool(pool)
     "
     ```
   - Update your best_bpb tracker.

8. **If val_bpb is equal or worse**:
   - `git reset` back to the previous commit.
   - Log the loss:
     ```
     python -c "
     from evo.crossover import load_pool, save_pool, log_loss
     pool = load_pool()
     pool = log_loss(pool, ('PARENT_A_ID', 'PARENT_B_ID'), NEW_BPB_OR_NONE)
     save_pool(pool)
     "
     ```

9. **Log to results.tsv** (same format as before, add parent info to description).

10. **Repeat from step 1.**

### Subagent prompt template

When spawning the crossover subagent, use a prompt like this (adapt based on parent types):

> Read these files:
> - Parent A ({type}): {content_path} — {summary}
> - Parent B ({type}): {content_path} — {summary}
> - Current code: train.py
>
> Write a new train.py that blends architectural ideas from both parents.
> Keep the code functional and runnable. It must:
> - Import from prepare.py: MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
> - Print val_bpb at the end
> - Respect the TIME_BUDGET for training duration
> - Not add new package dependencies
> - Stay in a single file
> - Update the MANIFEST comment block at the top
>
> Choose the most promising ideas from each parent. The result should be
> a coherent architecture, not a random mashup.

### Gene pool expansion

When the pool goes stale, search the web for a new architecture paper to inject. Staleness triggers:

- 5 consecutive losses, OR
- One entry holds >50% of total weight, OR
- All unique parent pairs in the pool have been tried (check history)

Prefer papers that introduce a novel mechanism and have high citation counts or top-venue publication. Ingest via:
```
python evo/extract_paper.py --arxiv <ID> --name "<short_name>"
```
Add at most one paper per staleness trigger.

### Fallback

If the gene pool has fewer than 2 entries, or if you want to try a purely intuition-driven experiment, you can skip the evolutionary protocol and edit `train.py` directly (the original approach). This is fine — not every experiment needs to come from crossover.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — sample new parent combinations from the gene pool, try crossing parents that haven't been combined before, or fall back to direct edits. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
