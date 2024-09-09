import argparse

def parse_arguments(arguments, flags):
    parser = argparse.ArgumentParser(
        prog='top',
        description='Show top lines from each file'
    )

    for opt_arg in arguments:
        parser.add_argument(opt_arg[0], type=opt_arg[1], default=None, help=opt_arg[2])

    for opt_flag in flags:
        parser.add_argument(opt_flag[0], action=opt_flag[1], default=None, help=opt_flag[2])

    args = parser.parse_args()

    return args