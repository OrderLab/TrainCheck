import argparse
import os
import signal
import subprocess
import time

# configs
$RAISE_SUBPROC_ERROR = True
os.environ["PYTHONUNBUFFERED"] = "1"

args = $ARGS
args = args[1:]

if "-o" in args or "--output" in args:
    idx = args.index("-o") if "-o" in args else args.index("--output")
    del args[idx:idx+2]

from traincheck.invariant import relation_pool
relation_names = [c.__name__ for c in relation_pool]

TMUX_SESSION_NAME = "run_infer_parallel"

def create_tmux_session():
    """If the tmux session does not exist, create it."""
    try:
        tmux has-session -t @(TMUX_SESSION_NAME)
    except subprocess.CalledProcessError:
        tmux new-session -d -s @(TMUX_SESSION_NAME)

largest_window_id = int($(tmux list-windows -t @(TMUX_SESSION_NAME) | awk '{print $1}' | sed 's/://g' | sort -n | tail -1).strip() or 0)

def run_cmd(cmd):
    global largest_window_id
    largest_window_id += 1
    tmux new-window -t @(TMUX_SESSION_NAME) -n @(largest_window_id)

    command = f"conda activate fp_torch222; python3 -m traincheck.infer_engine "
    command += " ".join(cmd)
    tmux send-keys -t @(TMUX_SESSION_NAME):@(largest_window_id) @(command) Enter

create_tmux_session()
# for relation in relation_names:
#     run_cmd(args + ["-o", f"inv_{relation}.json", "--enable-relation", relation])

run_cmd(args + ["-o", f"inv_FunctionCoverRelation.json", "--enable-relation", "FunctionCoverRelation"])
run_cmd(args + ["-o", f"inv_FunctionLeadRelation.json", "--enable-relation", "FunctionLeadRelation"])
run_cmd(args + ["-o", f"inv_other_relation.json", "--disable-relation", "FunctionCoverRelation", "FunctionLeadRelation"])