import os

from config import ROOT_DIR


def get_path_from_root(*subdirs):
    """
    Construct a path based on the root directory.

    :param subdirs: LIst of subdirectories or files; in order
    :return: full path from the root directory
    """
    return os.path.join(ROOT_DIR, *subdirs)

