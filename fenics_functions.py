import os
import inspect
from typing import TextIO
import sys
import shutil
import __main__
import resource
def save_current_script(base_folder: str):
    """
    Copy the user’s main script into base_folder/current_script_backup.py
    so you have an exact snapshot of what was run.
    """
    os.makedirs(base_folder, exist_ok=True)

    # 1) Try to get the path of the invoked script
    main_script = getattr(__main__, "__file__", None)
    if main_script is None:
        # Fallback: grab the caller’s filename instead
        caller_frame = inspect.stack()[1]
        main_script = caller_frame.filename

    # 2) Construct backup path
    script_name = os.path.basename(main_script)
    backup_path = os.path.join(base_folder, f"{script_name}.backup")

    # 3) Copy the script
    shutil.copy2(main_script, backup_path)

    print(f"Saved current script {main_script!r} to {backup_path!r}")


def start_logging(base_folder: str):
    """
    Redirect all console output (stdout & stderr) into
    base_folder/print_output.txt *and* still show it on screen.
    """
    class Tee:
        """
        Write to multiple file‐like streams at once.
        """
        def __init__(self, *streams: TextIO):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()
    
    # keep originals around if you ever want to restore:
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr    
    os.makedirs(base_folder, exist_ok=True)
    log_path = os.path.join(base_folder, "print_output.txt")
    log_file = open(log_path, "w")
    sys.stdout = Tee(_original_stdout, log_file)
    sys.stderr = Tee(_original_stderr, log_file)


def rss_mb():
    """Return resident‐set size (physical RAM) of this rank in MB."""
    # ru_maxrss is kB on Linux, bytes on macOS; divide by 1024 either way
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def report_rss(label, comm):
    """
    Gather RSS across ranks and print the max on rank-0 with a label.
    """
    rss_local = rss_mb()
    rss_all   = comm.gather(rss_local, root=0)

    print(f"[{label}]  peak RSS  = {max(rss_all):.1f}  MB")