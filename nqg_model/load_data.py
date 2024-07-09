
import pandas as pd # type: ignore
from nqg_model.types import *
from pathlib import Path
from typing import Optional
from nqg_model.helper import generate_users_file, generate_submissions_file
from nqg_model.types import IDMaps
from nqg_model.neural_quorum_governance import LAYER_1_NEURONS, LAYER_2_NEURONS, DEFAULT_NG_LAYERS
import numpy as np

def unknown_map(x: str, prefix: str='usr-') -> str:
    return f"{prefix}unknown-{str(x)[::3]}"

def retrieve_vote_data(folder_path: str, 
                       round_no: int, 
                       maps: IDMaps,
                       map_user_to_label=True,
                       map_project_to_label=True) -> tuple[set[ProjectUUID], set[UserUUID], ActionMatrix, VoteDecisionMatrix, NeuronPower, pd.DataFrame]:
    #extract round number from path

    vote_df = pd.read_csv(Path(folder_path) / "votes.csv")
    # if users.csv does not exist in folder_path, generate it
    if not (Path(folder_path) / 'users.csv').exists():
        generate_users_file(vote_df, folder_path)

    # if submissions.csv does not exist in folder_path, generate it
    if not (Path(folder_path) / 'submissions.csv').exists():
        generate_submissions_file(vote_df, folder_path)

    if map_user_to_label:
        vote_df['user'] = vote_df['user'].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))

    if map_project_to_label:
        vote_df['project'] = vote_df['project'].astype(str).map(lambda x: maps.SubmissionID2Label.get(x, unknown_map(x, "prj-")))

    columns = ["user", "project", "round", "vote_type", "delegation_result", "nqg_vote_power", "tally_vote_power"]

    projects = set(vote_df.project)
    users = set(vote_df.user)

    actions = {"Yes", "No", "Abstain", "Delegate"}

    # Initialize Action & Decision Matrix
    action_matrix: ActionMatrix = {}
    decision_matrix: VoteDecisionMatrix = {}

    # Create a list of layer names
    layer_names = [f'layer_{i}' for i in range(len(DEFAULT_NG_LAYERS))]

    # Create a list of neuron names for each layer
    neuron_names = [list(DEFAULT_NG_LAYERS[i][0].keys()) for i in range(len(DEFAULT_NG_LAYERS))]
    flat_neuron_names = [neuron for sublist in neuron_names for neuron in sublist]



    neuron_power_tensor: NeuronPower = xr.DataArray(
        data=np.full((len(users), len(projects), len(layer_names), len(flat_neuron_names)), np.nan),
        coords={
            'user': list(users),
            'project': list(projects),
            'layer': layer_names,
            'neuron': flat_neuron_names
        },
        dims=['user', 'project', 'layer', 'neuron'],
        attrs={'description': 'The power contribution by each neuron', 'long_name':'Neuron Power'}
    )

    for user in users:
        action_matrix[user] = {}
        decision_matrix[user] = {}
        for project in projects:
            action_matrix[user][project] = ProjectAction.Abstain
            decision_matrix[user][project] = Vote.Abstain

    for (i, row) in vote_df.iterrows():
        user = row.user
        project = row.project
        match row.vote_type:
            case "Delegate":
                action_matrix[user][project] = ProjectAction.Delegate
                decision_matrix[user][project] = Vote.Undefined
            case "Yes":
                action_matrix[user][project] = ProjectAction.Vote
                decision_matrix[user][project] = Vote.Yes
            case "No":
                action_matrix[user][project] = ProjectAction.Vote
                decision_matrix[user][project] = Vote.No
            case "Abstain":
                action_matrix[user][project] = ProjectAction.Abstain
                decision_matrix[user][project] = Vote.Abstain
            
    return (projects, users, action_matrix, decision_matrix, neuron_power_tensor, vote_df)



def get_delegation_graph(df: pd.DataFrame, maps: IDMaps) -> DelegationGraph:
    col ='user_delegator'
    col2 = 'user_delegate'
    df[col] = df[col].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    df[col2] = df[col2].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    delegation_graph: DelegationGraph = {}
    for i, group_df in df.groupby("user_delegator"):
        delegation_graph[i] = list(group_df.sort_values("rank").user_delegate.values) # type: ignore
        # NOTE: no checks for ranks are being made
    return delegation_graph


######
def get_trust_graph(df: pd.DataFrame, maps: IDMaps) -> TrustGraph:
    col ='user_trustee'
    col2 = 'user_trusted'
    df[col] = df[col].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    df[col2] = df[col2].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    trust_graph: TrustGraph = {}
    for i, group_df in df.groupby("user_trustee"):
        trust_graph[i] = set(group_df.user_trusted.values) # type: ignore
    return trust_graph

def get_reputations(df: pd.DataFrame, maps: IDMaps) -> UserReputations:
    col ='user'
    df[col] = df[col].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    reputations: UserReputations = {}
    for i, row in df.iterrows():
        match row.active_reputation_tier:
            case 0:
                reputations[row.user] = ReputationCategory.Verified
            case 1:
                reputations[row.user] = ReputationCategory.Pathfinder
            case 2:
                reputations[row.user]= ReputationCategory.Navigator
            case 3:
                reputations[row.user] = ReputationCategory.Pilot
            case _:
                reputations[row.user] = ReputationCategory.Unknown
    return reputations


def get_prior_voting(df: pd.DataFrame, maps: IDMaps) -> UserPriorVoting:
    col ='user'
    df[col] = df[col].astype(str).map(lambda x: maps.UserID2Wallet.get(x, unknown_map(x)))
    prior_voting: UserPriorVoting = {}
    for i, group_df in df.groupby("user"):
        prior_voting[i] = set(group_df.past_round.values) # type: ignore
    
    return prior_voting

def retrieve_other_data(folder_path: str, round_no: int, maps: IDMaps) -> tuple[DelegationGraph, TrustGraph, UserReputations, UserPriorVoting]:
    path = Path(folder_path)
    delegation_df = pd.read_csv(path / "delegation_graph.csv").query(f"round == {round_no}")
    trust_df = pd.read_csv(path / "trust_graph.csv").query(f"round == {round_no}")
    reputation_df = pd.read_csv(path / "user_reputation.csv").query(f"round == {round_no}")
    vote_history_df = pd.read_csv(path / "user_vote_history.csv").query(f"round == {round_no}")

    delegation_graph = get_delegation_graph(delegation_df, maps)
    trust_graph = get_trust_graph(trust_df, maps)
    reputations = get_reputations(reputation_df, maps)
    prior_voting = get_prior_voting(vote_history_df, maps)

    return (delegation_graph, trust_graph, reputations, prior_voting)

def retrieve_id_maps(folder_path: str) -> IDMaps:
    map_user_id_to_wallet = pd.read_csv(Path(folder_path) / 'users.csv').astype(str).set_index("_id")['public_key'].to_dict()
    map_user_id_to_handle = pd.read_csv(Path(folder_path) / 'users.csv').astype(str).set_index("_id")['username'].to_dict()
    map_submission_id_to_label = pd.read_csv(Path(folder_path) / 'submissions.csv').astype(str).set_index("_id")['name'].to_dict()

    maps = IDMaps(map_user_id_to_wallet, map_user_id_to_handle, map_submission_id_to_label)
    return maps