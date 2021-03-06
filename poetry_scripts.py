from subprocess import check_call
import sys
import re
import os


def test():
    check_call(["poetry", "run", "pytest", "--junit-xml=TEST-junit.xml", "-s"])


def release():
    tag = sys.argv[1]
    print("Releasing version", tag)
    version = _version_from_tag(tag)
    test_ci()
    check_call(["poetry", "version", version])
    check_call(["poetry", "build"])
    username = os.environ.get("PYPI_USERNAME")
    password = os.environ.get("PYPI_PASSWORD")
    check_call(["poetry", "publish", "-u", username, "-p", password])


def _version_from_tag(tag):
    """
    Extracts the semver from the given tag (e.g: v0.1.0 -> 0.1.0)
    :param tag:
    :return:
    """
    pattern = re.compile('v\d+.\d+.\d+')
    if not pattern.match(tag):
        raise Exception(f"{tag} is not valid semver")
    return tag[1:]