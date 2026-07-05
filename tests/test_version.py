import sys

sys.path.insert(0, ".")
try:
    from spectralstream import version
except ImportError:
    pass


def test_version_string():
    assert isinstance(version.__version__, str)
    assert len(version.__version__) > 0


def test_version_info():
    assert isinstance(version.__version_info__, tuple)
    assert len(version.__version_info__) == 3


def test_version_matches_info():
    major, minor, patch = version.__version_info__
    assert version.__version__ == f"{major}.{minor}.{patch}"
