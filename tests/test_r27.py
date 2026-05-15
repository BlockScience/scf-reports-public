import pandas as pd # type: ignore
import random
import numpy as np

from nqg_model.load_data import retrieve_other_data, retrieve_vote_data, retrieve_id_maps
from nqg_model.experiment import full_historical_data_run_plus_counterfactual
from nqg_model.load_data import retrieve_id_maps
from nqg_model.types import *

import xarray as xr

def run_r27_data():
    round_no = 27
    folder_path = f'data/r{round_no}/'

    (sim_backtest_df, df) = full_historical_data_run_plus_counterfactual(folder_path=folder_path, round_no=round_no)
    sim_df = pd.concat([sim_backtest_df], ignore_index=True)

    df.rename(columns={'user': 'user_ref', 'project': 'submission', 'vote_type': 'vote_type'}, inplace=True)

    return sim_df, df


def test_user_quorum_order_across_runs():

    N_loads = 5

    first_dict = None
    # user_quorum_across_load_timestep: dict[int, dict[int, DelegationGraph]] = {}

    for load in range(N_loads):
        sim_df, df = run_r27_data()
        sim_dict = sim_df.to_dict(orient='records')

        if load == 0:
            first_dict = sim_dict
        else:
            for i, record in enumerate(sim_dict):
                for column, value in record.items():
                    first_value = first_dict[i][column]

                    if type(value) == xr.DataArray:
                        xr.testing.assert_identical(value, first_value)
                    else:
                        np.testing.assert_equal(value, first_value, err_msg=f"Error at {load}, {i}, {column}")

    assert 1 == 1



def retrieve_data():
    round_no = 27
    folder_path = f'data/r{round_no}/'

    maps: IDMaps = retrieve_id_maps(folder_path)

    (N_timesteps, N_samples) = (1, 1)

    (projects, users_ids, action_matrix,
        decision_matrix, neuron_power_tensor, vote_df) = retrieve_vote_data(folder_path, round_no, maps, map_project_to_label=False)
    (delegates, trustees, user_reputations,
        user_past_votes) = retrieve_other_data(folder_path, round_no, maps)




    return [projects, users_ids, action_matrix, decision_matrix, delegates, trustees, user_reputations, user_past_votes, vote_df, neuron_power_tensor]

def test_retrieve_data_repeated():

    N_retrievals = 10

    first_data = None

    for i in range(N_retrievals):
        data = retrieve_data()

        if i == 0:
            first_data = data
        else:
            assert data[:-2] == first_data[:-2]
            pd.testing.assert_frame_equal(data[-2], first_data[-2])
            xr.testing.assert_identical(data[-1], first_data[-1])