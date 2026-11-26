# PMPP, worked through

**English** · [Tiếng Việt](README.vi.md)

An interactive study copy of *Programming Massively Parallel Processors* (5th edition),
built for one reader: lessons with simulations and narrated animation in a local web
app, and labs that run on your own GPUs through CUDA, HIP or Metal.

Status (2026-11-21): every chapter is built. That is chapters 1 to 24, appendices A and C
(B is skipped: deep learning basics), and three modules the book does not have: X1 sampling
kernels, X2 quantised kernels, X3 reading real backends. There are 79 labs, and they carry the
project milestones M0 to M10 (see `PLAN.md`, section 5). Chapter 5 was the pilot and the
template. `PROGRESS.md` holds the current state and what is still open; the plan is in
`PLAN.md`.

## Run the app

```sh
./start.sh          # then open http://localhost:5173
```

Needs Node.js 20 or newer. Your progress is saved to `progress/progress.json`; commit it
or sync it to carry it between machines.

## Run the labs

```sh
./verify.sh                         # build + test + benchmark everything this machine supports,
                                    # writes reports/verify-<time>-<os>-<backends>.md
./verify.sh --quick                 # same without benchmarks
./verify.sh --backend hip           # one backend only (repeatable), e.g. skip the slow emulators
python3 labs/pmpp.py doctor         # which backends were found, and what the device reports
python3 labs/pmpp.py run 5.2        # test your work for lab 5.2 on every backend
python3 labs/pmpp.py run 5.2 --bench
python3 labs/pmpp.py run 5.2 --backend emu
python3 labs/pmpp.py run 5.2 --solution
```

Needs CMake 3.21 or newer, a C++17 compiler and Python 3. Backends are detected and the
missing ones are skipped:

| Backend | Needs | Status of the reference solutions |
|---|---|---|
| `metal` | macOS with Xcode command line tools | verified on Metal (the author's Mac), except labs 11.5, 12.5 and 18.1 part B, which are CUDA-family only |
| `emu` | nothing extra | emulator-checked: a CPU emulator of the CUDA subset, logic only. Its test limit is 2400 s (900 s on devices; `PMPP_TEST_TIMEOUT` overrides), because lab X3.2's suite runs every thread in turn |
| `race` | nothing extra | the same emulator with threads run concurrently under ThreadSanitizer, for the labs that opt in |
| `cuda` | NVIDIA CUDA toolkit | front-end-checked; **never run on a CUDA device** |
| `hip` | AMD ROCm (`ROCM_PATH` or `/opt/rocm`) | **unverified**, never compiled |

If you have a CUDA or ROCm machine: run `./verify.sh` there and keep the report it
writes. It records build errors and test failures in enough detail to fix from.

## Layout

```
app/        the web app; app/src/chapters/chNN/ holds a chapter's lessons
content/    narration scripts, audio and manifests
labs/       build system, runtime, CPU emulator, and labs/chNN/labM_name/{starter,work,solution}
notes/      notes on every chapter of the book, with page references
progress/   your progress file
reports/    verify reports
```

The book's PDF is not part of the repository and is not needed to run anything.
