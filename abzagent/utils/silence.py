# abzagent/utils/silence.py
import os

def install_silence():
    # Reduce noisy native logs (Windows & other platforms)
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_LOG_SEVERITY", "fatal")
