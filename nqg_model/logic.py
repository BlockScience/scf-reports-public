from cadCAD_tools.types import Signal, VariableUpdate  # type: ignore
from nqg_model.types import *
from typing import Callable, Mapping
from copy import deepcopy
from scipy.stats import poisson, bernoulli  # type: ignore
from random import choice, sample
from nqg_model.neural_quorum_governance import *
from nqg_model.helper import *
import networkx as nx  # type: ignore


def generic_policy(_1, _2, _3, _4) -> dict:
    """Function to generate pass through policy

    Args:
        _1
        _2
        _3
        _4

    Returns:
        dict: Empty dictionary
    """
    return {}


def replace_suf(variable: str, default_value=0.0) -> Callable:
    """Creates replacing function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (variable, signal.get(variable, default_value))


def add_suf(variable: str, default_value=0.0) -> Callable:
    """Creates adding function for state update from string

    Args:
        variable (str): The variable name that is updated

    Returns:
        function: A function that continues the state across a substep
    """
    return lambda _1, _2, _3, state, signal: (variable, signal.get(variable, default_value) + state[variable])

def p_evolve_time(params: NQGModelParams, _2, _3, _4) -> TimestepSignal:
    """
    Generate a TimestepSignal based on the given model parameters.
    
    Parameters:
        params (NQGModelParams): The parameters of the NQG model which include the timestep duration.
        _2, _3, _4: Additional parameters that are not used in this function.
        
    Returns:
        TimestepSignal: A dictionary containing the change in days ('delta_days') based on the timestep duration.
    """
    return {'delta_days': params['timestep_in_days']}


def s_days_passed(_1, _2, _3, state: NQGModelState, signal: TimestepSignal) -> VariableUpdate:
    """
    Update the 'days_passed' state variable based on the TimestepSignal.
    
    Parameters:
        _1, _2, _3: Additional parameters that are not used in this function.
        state (NQGModelState): The current state of the NQG model.
        signal (TimestepSignal): The signal containing the change in days ('delta_days').
        
    Returns:
        VariableUpdate: A tuple containing the name of the variable to update ('days_passed') and its new value.
    """
    return ('days_passed', state['days_passed'] + signal['delta_days'])


def s_delta_days(_1, _2, _3, _4, signal: Signal) -> VariableUpdate:
    """
    Update the 'delta_days' state variable based on the provided signal.
    
    Parameters:
        _1, _2, _3, _4: Additional parameters that are not used in this function.
        signal (Signal): The signal containing the change in days ('delta_days').
        
    Returns:
        VariableUpdate: A tuple containing the name of the variable to update ('delta_days') and its new value.
    """
    return ('delta_days', signal['delta_days'])


def s_onboard_users(params: NQGModelParams, _2, _3, state: NQGModelState, _5) -> VariableUpdate:
    """
    Onboard N new users and their relevant properties for NQG
    through stochastic processes.

    XXX: the new user reputation is chosen from the `ReputationCategory` enum
    with every option having equal weight.
    XXX: the active past rounds for the new user is randomly sampled
    from the list of past rounds with equal weights. The amount of samples
    is based on a capped poisson sample.
    """
    new_user_list = deepcopy(state['users'])

    avg_new_users_per_ts = params['avg_new_users_per_day'] * \
        params['timestep_in_days']
    
    new_users: int = poisson.rvs(avg_new_users_per_ts)

    past_round_choices = params['past_rounds']
    reputation_choices = list(ReputationCategory)  # TODO: parametrize

    for i in range(new_users):
        past_voting_n = min(poisson.rvs(params['avg_user_past_votes']),
                            len(past_round_choices))

        new_user = User(label=str(len(new_user_list) + i),
                        reputation=choice(reputation_choices),
                        active_past_rounds=set(sample(list(past_round_choices), past_voting_n)))

        new_user_list.append(new_user)

    return ('users', new_user_list)


def p_user_vote(params: NQGModelParams,
                _2,
                history: list[list[NQGModelState]],
                state: NQGModelState) -> UserActionSignal:
    """
    Make new users decide on their actions: Abstain, Vote or Delegate

    XXX: Bernoulli processes are used for all of the following:
        - determining the probability of a user participating (actively or delegating) or not.
        - determine whatever the user will actively vote or delegate
        - determine if the user will vote on a project or not
        - determine if the user will vote yes or no on a project 
    XXX: Poisson processes are used for all of the following:
        - determine how much delegatees an user will have if he opted to delegate 
    
    Parameters:
        params (NQGModelParams): The parameters of the NQG model.
        _2: Additional parameter not used in this function.
        history (list[list[NQGModelState]]): The history of model states.
        state (NQGModelState): The current state of the NQG model.

    Returns:
        UserActionSignal: A dictionary containing updated 'delegatees', 'action_matrix', and 'vote_decision_matrix'.
    """
    delegates: DelegationGraph = deepcopy(state['delegatees'])
    action_matrix: ActionMatrix = deepcopy(state['action_matrix'])
    decision_matrix: VoteDecisionMatrix = deepcopy(state['vote_decision_matrix'])

    # Get the set of current users and previous users
    current_users = set(u.label
                        for u
                        in state['users'])

    previous_state_users: set[UserUUID] = retrieve_prev_state_users(history)

    # Identify new users by comparing the current state with the previous state
    if state['timestep'] > 1: # type: ignore
        new_users: set[UserUUID] = current_users - previous_state_users
    else:
        new_users = set()

    # Determining the actions of the new users
    for user in new_users:
        action_matrix[user] = {}
        decision_matrix[user] = {}

        for project in params['projects']:
            if bernoulli.rvs(params['new_user_action_probability']):
                if bernoulli.rvs(params['new_user_project_vote_probability']):
                    action_matrix[user][project] = ProjectAction.Vote
                    if bernoulli.rvs(params['new_user_project_vote_yes_probability']):
                        decision_matrix[user][project] = Vote.Yes
                    else:
                        decision_matrix[user][project] = Vote.No
                else:
                    action_matrix[user][project] = ProjectAction.Delegate
                    decision_matrix[user][project] = Vote.Undefined

                    # Assign delegates
                    mu: float = params['new_user_average_delegate_count'] - params['new_user_min_delegate_count']
                    delegate_count = poisson.rvs(mu, loc=params['new_user_min_delegate_count'])
                    delegate_count += params['new_user_min_delegate_count']
                    if delegate_count > len(previous_state_users):
                        delegate_count = len(previous_state_users)

                    if delegate_count < params['new_user_min_delegate_count']:
                        pass
                    else:
                        user_delegates: list[UserUUID] = sample(list(previous_state_users), delegate_count)
                        delegates[user] = user_delegates

            else:
                action_matrix[user][project] = ProjectAction.Abstain
                decision_matrix[user][project] = Vote.Abstain

    return {'delegatees': delegates,
            'action_matrix': action_matrix,
            'vote_decision_matrix': decision_matrix}


def s_trust(params: NQGModelParams, _2, history, state: NQGModelState, _5) -> VariableUpdate:
    """
    Make new users trust each other

    XXX: this is done by randomly sampling the set of previous users. The amount
    of users to be trusted is sampled from a Poisson distribution.


    Parameters:
        params (NQGModelParams): The parameters of the NQG model.
        _2: Additional parameter not used in this function.
        history: The history of model states.
        state (NQGModelState): The current state of the NQG model.
        _5: Additional parameter not used in this function.

    Returns:
        VariableUpdate: A tuple containing the name of the variable to update ('trustees') and its new value.
    """
    trustees: TrustGraph = deepcopy(state['trustees'])
    current_users: set[UserUUID] = {u.label
                                    for u
                                    in state['users']}

    previous_state_users: set[UserUUID] = retrieve_prev_state_users(history)

    # For making sense of having a initial state
    if state['timestep'] > 1: # type: ignore
        new_users: set[UserUUID] = current_users - previous_state_users
    else:
        new_users = set()
        
    # Identify new users by comparing the current state with the previous state
    for user in new_users:
        n_user_trustees = poisson.rvs(params['new_user_average_trustees'])
        n_user_trustees = min(n_user_trustees, len(previous_state_users))
        user_trustees: set[UserUUID] = set(
            sample(list(previous_state_users), n_user_trustees))
        trustees[user] = user_trustees

    return ('trustees', trustees)


def s_oracle_state(params: NQGModelParams, _2, _3, state: NQGModelState, _5) -> VariableUpdate:
    """
    Update the state of the oracles (eg. pagerank values & oracles/reputation weights)

    Parameters:
        params (NQGModelParams): The parameters of the NQG model.
        _2: Additional parameter not used in this function.
        _3: Additional parameter not used in this function.
        state (NQGModelState): The current state of the NQG model.
        _5: Additional parameter not used in this function.

    Returns:
        VariableUpdate: A tuple containing the name of the variable to update ('oracle_state') and its new value.
    """
    raw_graph = state['trustees']

    # Update Page rank values
    G = nx.from_dict_of_lists(raw_graph,
                              create_using=nx.DiGraph)
    pagerank_values: dict[UserUUID, float] = nx.pagerank(G,
                                                         alpha=0.85,
                                                         personalization=None,
                                                         max_iter=100,
                                                         tol=1e-6,
                                                         nstart=None,
                                                         dangling=None) # type: ignore

    # Update Reputation & Prior Voting user data

    reputation_values = {u.label: u.reputation for u in state['users']}
    prior_voting_values = {u.label: list(
        u.active_past_rounds) for u in state['users']}
    
    # Create a new OracleState with updated values
    new_state = OracleState(pagerank_results=pagerank_values,
                            reputation_bonus_values=reputation_values,
                            prior_voting_bonus_values=prior_voting_values,
                            reputation_bonus_map=state['oracle_state'].reputation_bonus_map,
                            prior_voting_bonus_map=state['oracle_state'].prior_voting_bonus_map)
    return ('oracle_state', new_state)


def p_compute_votes(params: NQGModelParams, _2, _3, state: NQGModelState) -> Signal:
    """
    Perform Neural Quorum Governance by computing the votes and updating the relevant matrices.

    Parameters:
        params (NQGModelParams): The parameters of the NQG model.
        _2: Additional parameter not used in this function.
        _3: Additional parameter not used in this function.
        state (NQGModelState): The current state of the NQG model.

    Returns:
        Signal: A dictionary containing the updated 'vote_power_matrix', 'vote_decision_matrix', 
                'per_project_voting', and 'neuron_power_tensor'.
    """
    action_matrix: ActionMatrix = deepcopy(state['action_matrix'])
    decision_matrix: VoteDecisionMatrix = deepcopy(state['vote_decision_matrix'])
    power_matrix: VotePowerMatrix = {}
    per_project_voting: PerProjectVoting = deepcopy(
        state['per_project_voting'])
    neuron_power_tensor: NeuronPower = deepcopy(state['neuron_power_tensor'])

    # Compute decision matrix with Quorum Delegation
    for user_id, actions in action_matrix.items():
        for project, action in actions.items():
            if action == ProjectAction.Delegate:
                if CounterfactualFlags.replace_delegate_by_abstain in params['counterfactual_flags']:
                    vote = Vote.Abstain
                else:
                    delegates = state['delegatees'].get(user_id, [])
                    vote = vote_from_quorum_delegation(delegates,
                                        project,
                                        state['action_matrix'],
                                        state['vote_decision_matrix'], 
                                        params)
                decision_matrix[user_id][project] = vote 

    # Compute vote matrix with Neural Governance
    for user_id, decisions in decision_matrix.items():
        power_matrix[user_id] = {}
        for project, decision in decisions.items():
            decision = decision_matrix[user_id][project]
            
            # Set power to 1 if counterfactual flag is set to replace neuron governance by one vote one power
            if CounterfactualFlags.replace_neuron_governance_by_one_vote_one_power in params['counterfactual_flags']:    
                power = 1.0
                
            else:
                power, neuron_power_dict = power_from_neural_governance(user_id,
                                                    project,
                                                    params['neuron_layers'],
                                                    state['oracle_state'],
                                                    params['initial_power'])
            power_matrix[user_id][project] = decision * power
            
            # Update neuron power tensor

            # Set neuron power to 1 if counterfactual flag is set to replace neuron governance by one vote one power
            if CounterfactualFlags.replace_neuron_governance_by_one_vote_one_power in params['counterfactual_flags']:  
                layer = neuron_power_tensor.loc[{'user': user_id, 'project': project}]._indexes['layer'].index[0]
                neuron = neuron_power_tensor.loc[{'user': user_id, 'project': project}]._indexes['neuron'].index[0]
                neuron_power_tensor.loc[{'user': user_id, 'project': project, 'layer': layer, 'neuron': neuron}] = power

            # Otherwise, update neuron power tensor with the power from the neuron power dictionary
            else:
                for (layer, neuron), power in neuron_power_dict.items():
                    neuron_power_tensor.loc[{'user': user_id, 'project': project, 'layer': layer, 'neuron': neuron}] = power

            # Update per project voting
            if project in per_project_voting:
                per_project_voting[project] += power_matrix[user_id][project]
            else:
                per_project_voting[project] = power_matrix[user_id][project]

    return {'vote_power_matrix': power_matrix,
            'vote_decision_matrix': decision_matrix,
            'per_project_voting': per_project_voting,
            'neuron_power_tensor': neuron_power_tensor}
