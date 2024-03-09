import os
import io
import json


def get_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


def get_files(directory):
    return [name for name in os.listdir(directory)]


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jlload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jldict = []
    for line in f:
        jldict.append(json.loads(line))
    return jldict
