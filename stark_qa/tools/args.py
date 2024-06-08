import argparse


def load_args(args_dict):
    """
    Convert a dictionary into an argparse.Namespace object.

    Args:
        args_dict (dict): Dictionary of arguments to be converted.

    Returns:
        argparse.Namespace: Namespace object with the arguments.
    """
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args


def merge_args(args_1, args_2):
    """
    Merge two argparse.Namespace objects. Arguments from args_2 have higher priority.

    Args:
        args_1 (argparse.Namespace): First namespace object.
        args_2 (argparse.Namespace): Second namespace object.

    Returns:
        argparse.Namespace: Merged namespace object.
    """
    args = argparse.Namespace()
    for key, value in args_1.__dict__.items():
        setattr(args, key, value)
    for key, value in args_2.__dict__.items():
        setattr(args, key, value)
    return args
