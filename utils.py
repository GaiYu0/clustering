import argparse
import shlex

def Parse(parser):
    class _Parse(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)

        def __call__(self, _, namespace, values, option_string=None):
            args = parser.parse_args(shlex.split(values))
            setattr(namespace, self.dest, args)

    return _Parse

sgd_parser = argparse.ArgumentParser()
sgd_parser.add_argument('--lr', type=float)
ParseSGDArgs = Parse(sgd_parser)

adam_parser = argparse.ArgumentParser()
adam_parser.add_argument('--lr', type=float)
adam_parser.add_argument('--amsgrad', type=bool)
ParseAdamArgs = Parse(adam_parser)
