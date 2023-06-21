from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-r",
    "--rooms",
    choices=[2, 4],
    default=2,
    type=int,
    help="Number of rooms for the simulation",
)
