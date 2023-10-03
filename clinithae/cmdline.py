# coding: utf8

import click

from clinithae.cli.train_cli import cli as train_cli
from clinithae.utils.logger import setup_logging

CONTEXT_SETTINGS = dict(
    # Extend content width to avoid shortening of pipeline help.
    max_content_width=160,
    # Display help string with -h, in addition to --help.
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Verbosity mode.")
def cli(verbose):
    """ClinicaDL command line.

    For more information please read the doc: https://clinicadl.readthedocs.io/en/latest/ .
    Source code is available on GitHub: https://github.com/aramis-lab/clinicaDL .

    Do not hesitate to create an issue to report a bug or suggest an improvement.
    """
    setup_logging(verbose=verbose)


cli.add_command(train_cli)

if __name__ == "__main__":
    cli()
