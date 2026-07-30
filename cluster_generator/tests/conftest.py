"""
Configure pytest for cluster_generator
"""

import os

import pytest

from cluster_generator.utils import cgparams

# Disable progress bars during tests -> GH actions cannot emulate console,
# prints each update on seperate line (slow).
cgparams["system"]["display"]["progress_bars"] = False


def pytest_collection_modifyitems(session, config, items):
    # Ensuring the order of necessary tests
    # --------------------------------------#
    # This is done to avoid having to rebuild toy-models for every
    # test when one is already generated. As such, we run tests on
    # ``test_model`` first to ensure that a model is generated and
    # saved to disk.
    # _enforced_test_ordering can be used to add order ensured tests
    # which are then forced to run ahead of the other tests.
    _enforced_test_ordering = [("cluster_generator.tests.test_model", "test_model_build")]

    _doc_tests, _standard_tests = (
        [test for test in items if isinstance(test, pytest.DoctestItem)],
        [test for test in items if not isinstance(test, pytest.DoctestItem)],
    )

    _test_dictionary = {(test.module.__name__, test.name): test for test in _standard_tests}

    items[:] = (
        [
            _test_dictionary.pop(test_name)
            for test_name in _enforced_test_ordering
            if test_name in _test_dictionary
        ]
        + list(_test_dictionary.values())
        + _doc_tests
    )
    print([k.name for k in items])


def pytest_addoption(parser):
    parser.addoption("--answer_dir", help="Directory where answers are stored.")
    parser.addoption(
        "--answer_store",
        action="store_true",
        help="Generate new answers, but don't test.",
    )
    parser.addoption("--tmp", help="The temporary directory to use.", default=None)


@pytest.fixture()
def answer_store(request) -> bool:
    """fetches the ``--answer_store`` option."""
    return request.config.getoption("--answer_store")


@pytest.fixture()
def answer_dir(request) -> str:
    """fetches the ``--answer_dir`` option."""
    ad = os.path.abspath(request.config.getoption("--answer_dir"))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad


@pytest.fixture()
def temp_dir(request) -> str:
    """
    Pull the temporary directory. If this is specified by the user,
    then it may be a non-temp directory which is not
    wiped after runtime. If not specified, then a temp directory is
    generated and wiped after runtime.
    """
    td = request.config.getoption("--tmp")

    if td is None:
        from tempfile import TemporaryDirectory

        td = TemporaryDirectory()

        yield td.name

        td.cleanup()
    else:
        yield os.path.abspath(td)
