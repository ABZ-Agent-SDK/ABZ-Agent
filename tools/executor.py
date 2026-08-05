# abzagent/tools/executor.py
import shlex
import subprocess

def run_command(command):
    try:
        # Replace 'build' with actual build command if needed
        if "build" in command.lower():
            cmd = ["echo", "Simulated build command"]
        else:
            cmd = shlex.split(command)
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        return result.stdout or result.stderr
    except Exception as e:
        return f"Error running command: {e}"
