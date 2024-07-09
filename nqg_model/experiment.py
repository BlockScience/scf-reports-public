from pandas import DataFrame  # type: ignore
from nqg_model.params import SINGLE_RUN_PARAMS, INITIAL_STATE
from nqg_model.types import *
from nqg_model.structure import NQG_MODEL_BLOCKS
import pandas as pd
from nqg_model.load_data import retrieve_other_data, retrieve_vote_data, retrieve_id_maps
from pathlib import Path
from cadCAD.tools import easy_run  # type: ignore

from nqg_model.types import IDMaps  # type: ignore


def full_historical_data_run_plus_counterfactual(folder_path: str,
                                                 round_no: int,
                                                 map_project_to_label=False) -> tuple[DataFrame,DataFrame]:
    """Function which runs the cadCAD simulations

    Returns:
        tuple[DataFrame, DataFrame]: A tuple containing two dataframes of simulation data and vote data
    """
    PAST_ROUNDS = set(i for i in range(1, round_no + 1))
    maps: IDMaps = retrieve_id_maps(folder_path)

    (N_timesteps, N_samples) = (1, 1)

    (projects, users_ids, action_matrix,
        decision_matrix, neuron_power_tensor, vote_df) = retrieve_vote_data(folder_path, round_no, maps, map_project_to_label=map_project_to_label)
    (delegates, trustees, user_reputations,
        user_past_votes) = retrieve_other_data(folder_path, round_no, maps)


    LABELS = ['backtesting', 'no_QD', 'no_NG', 'no_NQG',
              'without_first_layer', 'without_second_layer']

    FLAGS = [CounterfactualFlags.none,
             CounterfactualFlags.replace_delegate_by_abstain,
             CounterfactualFlags.replace_neuron_governance_by_one_vote_one_power,
             CounterfactualFlags.replace_delegate_by_abstain | CounterfactualFlags.replace_neuron_governance_by_one_vote_one_power,
             CounterfactualFlags.remove_first_neuron_layer | CounterfactualFlags.replace_delegate_by_abstain,
             CounterfactualFlags.remove_second_neuron_layer | CounterfactualFlags.replace_delegate_by_abstain,
             ]

    params = SINGLE_RUN_PARAMS.copy()
    params['label'] = LABELS  # type: ignore
    params['projects'] = projects
    params['counterfactual_flags'] = FLAGS  # type: ignore
    params['past_rounds'] = PAST_ROUNDS

    users = [User(uid,
                  user_reputations.get(uid, ReputationCategory.Unknown),
                  user_past_votes.get(uid, set()))  # type: ignore
             for uid in users_ids]

    INITIAL_ORACLE_STATE = OracleState(
        pagerank_results={},
        reputation_bonus_map={
            ReputationCategory.Pilot: 0.3,
            ReputationCategory.Navigator: 0.2,
            ReputationCategory.Pathfinder: 0.1,
            ReputationCategory.Verified: 0.0,
            ReputationCategory.Unknown: 0.0},
        prior_voting_bonus_map={round: 0.1 for round in PAST_ROUNDS},
        reputation_bonus_values={},
        prior_voting_bonus_values={}
    )

    initial_state = INITIAL_STATE.copy()
    initial_state['oracle_state'] = INITIAL_ORACLE_STATE
    initial_state['users'] = users
    initial_state['action_matrix'] = action_matrix
    initial_state['vote_decision_matrix'] = decision_matrix
    initial_state['delegatees'] = delegates
    initial_state['trustees'] = trustees
    initial_state['neuron_power_tensor'] = neuron_power_tensor

    param_dict: dict[str, list] = {k: [p] for k, p in params.items() if type(
        p) != list or k == 'neuron_layers'}
    param_dict = {**params, **param_dict}  # type: ignore

    # Load simulation arguments
    sim_args = (initial_state,
                param_dict,
                NQG_MODEL_BLOCKS,
                N_timesteps,
                N_samples)

    # Run simulation
    sim_df = easy_run(*sim_args, assign_params={'label'}, exec_mode='single')
    return (sim_df, vote_df)
