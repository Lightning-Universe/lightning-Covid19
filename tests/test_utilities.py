import os

from detect_covid19 import PATH_ROOT


def test_paths():
    assert os.path.isdir(PATH_ROOT)
