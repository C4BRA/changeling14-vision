# Run folder contract

Every pipeline step (import replays, build dataset, train, benchmark, export)
can report what it is doing by writing a few files into a **run folder**.
The training app (C4BRA/changeling14-training-gui) reads these files and never
writes the step's own outputs. A step that runs without the app behaves the
same; it just has nobody watching.

A step finds its run folder in the `C14_RUN_DIR` environment variable, or
`--run-dir`. If neither is set, it writes no run files.

## Layout

```
<runs root>/<YYYYMMDD-HHMMSS>-<job id>/
  run.json        written by the app's job supervisor, never by the step
  console.log     the step's stdout and stderr, captured by the supervisor
  status.json     written by the step: current progress
  watchset/       written by the step: images to look at, grouped by checkpoint
    <checkpoint>/
      meta.json   optional: {"step": 1200, "time": 1792680000.0, ...}
      *.png
  ...             anything else the step wants (metrics, reports)
```

## status.json

Rewritten in place (write a temp file, then rename) whenever progress moves,
at most a few times a second.

| Field         | Type   | Meaning                                                  |
|---------------|--------|----------------------------------------------------------|
| `stage`       | string | Short name of the current phase, e.g. `cut sprites`      |
| `step`        | int    | Units done so far in this stage                           |
| `total`       | int    | Units in this stage, if known                             |
| `message`     | string | One line for people, e.g. `loss 0.412`                    |
| `eta_seconds` | number | Optional. The step's own estimate of time left overall    |
| `updated_at`  | number | Unix time of this write                                   |

All fields are optional. Without `eta_seconds` the app estimates time left
from `step / total` and the time since the run started.

## watchset/

One subfolder per checkpoint. The app sorts them by `meta.json` `step` when
present, otherwise by folder name, and shows them on a timeline slider.
Images are drawn at native pixel size and only ever scaled up by whole
numbers, so pixel art stays crisp. Training writes the fixed watch-set frames
with predictions drawn on; other steps may use it for a sample gallery.

## run.json

Written by the supervisor that starts the step. Steps should not touch it.

| Field        | Meaning                                                     |
|--------------|-------------------------------------------------------------|
| `job`        | Job id from the app's jobs config                           |
| `label`      | Button label                                                |
| `heavy`      | Whether the job takes the one heavy slot                    |
| `command`    | The command line that was run                               |
| `state`      | `queued`, `running`, `done`, `failed`, `stopped`, `cancelled` |
| `pid`        | Supervisor process id (also its process group)              |
| `queued_at`, `started_at`, `ended_at` | Unix times                        |
| `exit_code`  | The step's exit code once it ends                           |

## Stopping

The app stops a job by sending SIGTERM to its process group. A step should
finish what it is writing, leave its outputs consistent, and exit within about
15 seconds; after that it is killed.
