# abzagent/tools/file_ops.py
import os
from pathlib import Path

def handle_file_command(command: str) -> str:
    """
    Minimal file editor simulation.
    """
    # Example: parse "update main.py add print('hi')"
    try:
        words = command.split()
        if "create" in words:
            idx = words.index("create")
            filename = words[idx + 1]
            Path(filename).write_text("# New file created by ABZ Agent\n")
            return f"✅ Created file: {filename}"
        elif "update" in words or "edit" in words or "fix" in words:
            # Simplified: append command as comment
            filename = words[-1]  # assume last word is filename
            if not os.path.exists(filename):
                return f"❌ File does not exist: {filename}"
            with open(filename, "a") as f:
                f.write(f"# Edited by ABZ Agent: {command}\n")
            return f"✅ Updated file: {filename}"
        else:
            return "🤖 No valid file action detected."
    except Exception as e:
        return f"❌ Error handling file: {e}"
