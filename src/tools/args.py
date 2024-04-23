import argparse


def load_args(dict):
    args = argparse.Namespace()
    for key, value in dict.items():
        setattr(args, key, value)
    return args


def merge_args(args_1, args_2):
    # merge two namespaces into args, args_2 has higher priority
    args = argparse.Namespace()
    for key, value in args_1.__dict__.items():
        setattr(args, key, value)
    for key, value in args_2.__dict__.items():
        setattr(args, key, value)

    return args