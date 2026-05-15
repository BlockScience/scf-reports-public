from nqg_model.types import *
from functools import reduce

# Part 1. General definitions


def vote_from_quorum_delegation(user_quorum: list[UserUUID],
                                project_id: ProjectUUID,
                                action_matrix: ActionMatrix,
                                decision_matrix: VoteDecisionMatrix,
                                params: NQGModelParams) -> Vote:
    """
    Compute the quorum agreement for the active participants.

    Parameters:
        user_quorum (list[UserUUID]): List of user IDs forming the quorum.
        project_id (ProjectUUID): The project ID for which the vote is being computed.
        action_matrix (ActionMatrix): Dictionary containing user actions for each project.
        decision_matrix (VoteDecisionMatrix): Dictionary containing user votes for each project.
        params (NQGModelParams): The parameters of the NQG model.

    Returns:
        Vote: The resolved vote based on quorum consensus (Yes, No, or Abstain).
    """
    # Filter User quorum for actively voting users only.
    valid_delegates = [u
                       for u, a in action_matrix.items()
                       if a[project_id] == ProjectAction.Vote
                       and u in user_quorum]

    # Select up to the max quorum selected delegates parameter
    if len(valid_delegates) > params['max_quorum_selected_delegates']:
        selected_delegates = valid_delegates[:
                                             params['max_quorum_selected_delegates']]
    else:
        selected_delegates = valid_delegates
        

    # Compute Quorum Agreement and Size.
    agreement = 0.0
    quorum_size = 0
    for delegate in selected_delegates:
        delegate_decisions: dict[ProjectUUID, Vote] = decision_matrix.get(delegate, {})
        action: Vote | None = delegate_decisions.get(project_id, None)
        if action == Vote.Yes or action == Vote.No:
            quorum_size += 1
            if action is Vote.Yes:
                agreement += params['quorum_agreement_weight_yes']
            elif action is Vote.No:
                agreement += params['quorum_agreement_weight_no']
            else:
                agreement += params['quorum_agreement_weight_abstain']

    # Compute Absolute and Relative agreement fractions
    absolute_agreement = agreement / params['max_quorum_selected_delegates']
    if quorum_size > 0:
        relative_agreement = agreement / quorum_size
    else:
        relative_agreement = 0.0

    # Resolve vote as per quorum consensus
    resolution = Vote.Abstain

    if len(valid_delegates) >= params['min_quorum_threshold']:
        if abs(absolute_agreement) >= params['quorum_delegation_absolute_threshold']:
            if abs(relative_agreement) >= params['quorum_delegation_relative_threshold']:
                if relative_agreement > 0:
                    resolution = Vote.Yes
                elif relative_agreement < 0:
                    resolution = Vote.No
                else:
                    resolution = Vote.Abstain
            else:
                resolution = Vote.Abstain
        else:
            resolution = Vote.Abstain
    else:
        resolution = Vote.Abstain


    return resolution


def power_from_neural_governance(uid: UserUUID,
                                 pid: ProjectUUID,
                                 neuron_layers: list[NeuronLayer],
                                 oracle_state: OracleState,
                                 initial_votes: float = 0.0,
                                 print_on_each_layer=False,
                                 print_on_each_neuron=False) -> tuple[VotingPower, dict[tuple[str,str], float]]:
    """
    Computes a User Vote towards a Project as based on 
    a Feedforward implementation of Neural Governance for a strictly
    sequential network (no layer parallelism).

    Parameters:
    uid (UserUUID): The user ID.
    pid (ProjectUUID): The project ID.
    neuron_layers (list[NeuronLayer]): List of neuron layers, each containing a dictionary of neurons 
                                        and a layer aggregator function.
    oracle_state (OracleState): The current state of the oracle.
    initial_votes (float): The initial vote value to start the computation. Default is 0.0.
    print_on_each_layer (bool): If True, prints the vote value after each layer. Default is False.
    print_on_each_neuron (bool): If True, prints the neuron power for each neuron. Default is False.

    Returns:
        tuple[VotingPower, dict[tuple[str, str], float]]: 
            - The final voting power after passing through all neuron layers.
            - A dictionary mapping each neuron (identified by layer and label) to its computed power.
    """
    current_vote = initial_votes
    if print_on_each_layer:
        print(f"Layer {0}: {current_vote}")
        
    neuron_power_dict = {}

    # Process each layer sequentially    
    for i, layer in enumerate(neuron_layers):
        (neurons, layer_aggregator) = layer
        neuron_votes = []

        # Process each neuron in the layer
        for (neuron_label, neuron) in neurons.items():
            (oracle_function, weighting_function) = neuron
            raw_neuron_vote = oracle_function(
                uid, pid, current_vote, oracle_state)
            neuron_power = weighting_function(raw_neuron_vote) 

            if print_on_each_neuron:
                print(f"{uid}, {pid}, {i}, {neuron_label}, {neuron_power}")
            neuron_votes.append(neuron_power)
            neuron_power_dict[(f'layer_{i}',neuron_label)] = neuron_power

        # Aggregate neuron votes to get the new vote
        new_vote = layer_aggregator(neuron_votes)
        current_vote = new_vote

        if print_on_each_layer:
            print(f"Layer {i+1}: {current_vote}")

    return (current_vote, neuron_power_dict)

# Part 2. Specific definitions
# Prior Voting Bonus


def prior_voting_score(user_id: UserUUID, oracle_state: OracleState, past_layer_power, init_bonus=0.0) -> VotingPower:
    """
    Compute the prior voting score for a user.

    This function calculates the prior voting score for a user by summing the
    initial bonus with the bonuses from the user's prior voting history.

    Parameters:
        user_id (UserUUID): The unique identifier for the user.
        oracle_state (OracleState): The current state of the oracle, containing prior voting bonus values and maps.
        past_layer_power (VotingPower): The voting power from the previous layer.
        init_bonus (float): The initial bonus to be added to the prior voting score. Default is 0.0.

    Returns:
        VotingPower: The computed prior voting score.
    
    Implementation source:
    https://github.com/BlockScience/stellar-community-fund-extras/blob/c66b850af42bccddb59127a915bec4835d934ae7/voting/src/voting_system/src/neurons/prior_voting_history_neuron.rs#L18
    """
    bonus = init_bonus
    for r in oracle_state.prior_voting_bonus_values[user_id]:
        bonus += oracle_state.prior_voting_bonus_map.get(r, 0.0)
        
    return bonus + past_layer_power


# Reputation Bonus

def reputation_score(user_id: UserUUID, oracle_state: OracleState) -> VotingPower:
    """
    Compute the reputation score for a user.

    This function calculates the reputation score for a user based on their reputation category,
    using the reputation bonus map provided in the oracle state.

    Parameters:
        user_id (UserUUID): The unique identifier for the user.
        oracle_state (OracleState): The current state of the oracle, containing reputation bonus values and maps.

    Returns:
        VotingPower: The computed reputation score.
    """
    return oracle_state.reputation_bonus_map.get(oracle_state.reputation_bonus_values[user_id], 0.0)

# Trust Bonus


def trust_score(user_id: UserUUID, 
                oracle_state: OracleState, 
                norm=True,
                non_norm_scale_factor=178) -> VotingPower:
    """
    Compute the Trust Score based on the Canonical PageRank.

    This function calculates the trust score of a user by computing the PageRank on the 
    entire trust graph and scaling the results using Min-Max normalization or a non-normalized 
    scale factor.

    Parameters:
        user_id (UserUUID): The unique identifier for the user.
        oracle_state (OracleState): The current state of the oracle, containing PageRank results.
        norm (bool): If True, applies Min-Max normalization to scale the trust scores between 0.0 and 1.0. 
                     If False, scales the raw PageRank value using the non_norm_scale_factor. Default is True.
        non_norm_scale_factor (int): The scale factor to apply if normalization is not used. Default is 178.

    Returns:
        VotingPower: The computed trust score.

    Implementation source:
    https://github.com/BlockScience/stellar-community-fund-extras/blob/c66b850af42bccddb59127a915bec4835d934ae7/voting/src/voting_system/src/neurons/trust_graph_neuron.rs#L9
    """
    pagerank_values: dict[UserUUID, float] = oracle_state.pagerank_results
    
    if (len(pagerank_values)) < 2 or (user_id not in pagerank_values.keys()):
        trust_score = 0.0
    else:
        raw_value: float = pagerank_values[user_id] # type: ignore

        if norm is True:
            max_value: float = max(pagerank_values.values())
            min_value: float = min(pagerank_values.values())
            if max_value == min_value:
                # XXX: assumption for edge cases
                trust_score = 0.5
            else:
                trust_score = (raw_value - min_value) / (max_value - min_value)
        else:
            trust_score = raw_value * non_norm_scale_factor
    return trust_score



def raw_trust_score(user_id: UserUUID, oracle_state: OracleState) -> VotingPower:
    """
    Retrieve the raw trust score based on PageRank results.

    Parameters:
        user_id (UserUUID): The unique identifier for the user.
        oracle_state (OracleState): The current state of the oracle, containing PageRank results.

    Returns:
        VotingPower: The raw trust score.
    """
    return oracle_state.pagerank_results.get(user_id, 0.0)

# Layering it together

# Aggregator functions
def SUM_AGGREGATOR(lst):
    """
    Sum aggregator function for neural layers.

    Parameters:
        lst (list[float]): List of voting powers.

    Returns:
        float: The sum of the voting powers.
    """
    return sum(lst)


def PRODUCT_AGGREGATOR(lst):
    """
    Product aggregator function for neural layers.

    Parameters:
        lst (list[float]): List of voting powers.

    Returns:
        float: The product of the voting powers.
    """
    return reduce((lambda x, y: x * y), lst)



LAYER_1_NEURONS = {
    'trust_score': (lambda u, p, x, s: trust_score(u, s),
                    lambda x: x),
    'reputation_score': (lambda u, p, x, s: reputation_score(u, s),
                         lambda x: x)
}

LAYER_1_ALT_NEURONS = LAYER_1_NEURONS.copy()
LAYER_1_ALT_NEURONS['trust_score'] = (lambda u, p, x, s: trust_score(u, s, True, 176),
                    lambda x: x)

LAYER_2_NEURONS = {
    'past_round': (lambda u, p, x, s: prior_voting_score(u, s, x, init_bonus=0.0),
                   lambda x: x)
}

# DEFAULT_NG_LAYERS: list[NeuronLayer] = [(LAYER_1_NEURONS, SUM_AGGREGATOR),
#                                         (LAYER_2_NEURONS, PRODUCT_AGGREGATOR)] 

# Definition of default neural governance layers

DEFAULT_NG_LAYERS: list[NeuronLayer] = [(LAYER_1_ALT_NEURONS, SUM_AGGREGATOR),
                                        (LAYER_2_NEURONS, PRODUCT_AGGREGATOR)] 

