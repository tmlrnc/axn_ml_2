import sys
import importlib
import pkgutil

from ohe.predictor import algorithm_lookup

def import_submodules(package_name):
    """ Import all submodules of a module, recursively
    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]
    """
    package = sys.modules[package_name]
    return {
        name: importlib.import_module(package_name + '.' + name)
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__)
    }


def get_algorithm_from_string(command_line_arg):
    if command_line_arg not in algorithm_lookup:
        raise Exception(f"No algorithm found for {command_line_arg}.")
    return algorithm_lookup[command_line_arg]

import_submodules('ohe.predictor.algorithm')