from nqg_model import default_run_args
from nqg_model.experiment import *
from cadCAD_tools.execution import easy_run # type: ignore
from datetime import datetime
import click
import os

#############
round_no = 27
#############

@click.command()
@click.option('-e', '--experiment-run', 'experiment_run',
              default=False,
              is_flag=True,
              help="Make an experiment run instead")
@click.option('-p', '--pickle', 'pickle', default=False, is_flag=True)
def main(experiment_run: bool, pickle: bool) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_run is True:
        df = easy_run(*default_run_args, assign_params=False)
    else:
        df, vote_df = full_historical_data_run_plus_counterfactual(folder_path=f'../data/r{round_no}', round_no=round_no)
    if pickle:
        df.to_pickle(
            f"data/simulations/multi-run-{timestamp}.pkl.gz", compression="gzip")


if __name__ == "__main__":
    main()