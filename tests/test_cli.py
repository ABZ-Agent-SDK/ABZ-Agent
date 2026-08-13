"""
Regression test for `abz-agents run <file>`: the script's own directory must
land on sys.path so sibling-module imports (e.g. `from tools import ...`)
work, matching what `python file.py` does automatically. runpy.run_path()
does not do this on its own — see abzagent/cli.py's _cmd_run.
"""
import os
import sys

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent.cli import _cmd_run


def test_run_supports_sibling_module_imports(tmp_path, capsys):
    (tmp_path / "helper.py").write_text("VALUE = 'from helper'\n", encoding="utf-8")
    main_path = tmp_path / "main.py"
    main_path.write_text(
        "from helper import VALUE\nprint(VALUE)\n", encoding="utf-8"
    )

    script_dir = str(tmp_path.resolve())
    try:
        exit_code = _cmd_run(str(main_path))
    finally:
        if script_dir in sys.path:
            sys.path.remove(script_dir)
        sys.modules.pop("helper", None)

    assert exit_code == 0
    assert "from helper" in capsys.readouterr().out


def test_run_missing_file_returns_error_code(capsys):
    exit_code = _cmd_run("does_not_exist.py")
    assert exit_code == 2
    assert "File not found" in capsys.readouterr().err
