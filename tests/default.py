from nqg_model.types import *
from nqg_model.neural_quorum_governance import DEFAULT_NG_LAYERS


# Constants defining the number of timesteps and samples
TIMESTEPS = 100
SAMPLES = 1

# Initial number of users and projects in the simulation
N_INITIAL_USERS = 6
N_PROJECTS = 15

# Number of past rounds used in the model
# NOTE: Up to date as per SCF #24
N_PAST_ROUNDS = 26

# Average number of past votes per user
AVERAGE_PAST_VOTES_PER_USER = 1.5
# Set of indices representing the past rounds
PAST_ROUNDS = set(i for i in range(N_PAST_ROUNDS))

# Set of default project identifiers
DEFAULT_PROJECTS = set(f"proj_{i}" for i in range(N_PROJECTS))

# Initial state of the Oracle, including PageRank results and bonus mappings
# NOTE: Up to date as per SCF #24
INITIAL_ORACLE_STATE = OracleState(
    pagerank_results={},  # Empty initially, will be populated during the simulation
    reputation_bonus_map={
        ReputationCategory.Pilot: 0.3,
        ReputationCategory.Navigator: 0.2,
        ReputationCategory.Pathfinder: 0.1,
        ReputationCategory.Verified: 0.0,
        ReputationCategory.Unknown: 0.0
    },
    prior_voting_bonus_map={round: 0.1 for round in PAST_ROUNDS},  # Uniform bonus for prior voting
    reputation_bonus_values={}, 
    prior_voting_bonus_values={}  
)


DEFAULT_NQG_PARAMS_FOR_TESTING = NQGModelParams(
    label='default_run',  # Label for this run
    timestep_in_days=1.0,  # Time step duration in days
    counterfactual_flags=CounterfactualFlags.none,  # No counterfactual scenarios enabled

    # Quorum Delegation Parameters
    # NOTE: Up to date as per SCF #24
    quorum_agreement_weight_yes=1.0,
    quorum_agreement_weight_no=-1.0,
    quorum_agreement_weight_abstain=0.0,
    min_quorum_threshold=2,
    max_quorum_selected_delegates=8,
    max_quorum_candidate_delegates=15,
    quorum_delegation_absolute_threshold=1/2,
    quorum_delegation_relative_threshold=2/3,

    neuron_layers=DEFAULT_NG_LAYERS,  # Default neuron layers from imported module

    initial_power=0.0,  # Initial power for neurons
    past_rounds=PAST_ROUNDS,  # Set of past rounds indices
    projects=DEFAULT_PROJECTS,  # Set of project identifiers

    # Behavioral Parameters (not used for backtesting / counterfactual scenarios)
    avg_new_users_per_day=1.0,  # Average number of new users per day
    avg_user_past_votes=AVERAGE_PAST_VOTES_PER_USER,  # Average past votes per user
    new_user_action_probability=0.5,  # Probability of new user taking an action
    new_user_project_vote_probability=5/N_PROJECTS,  # Probability of new user voting on a project
    new_user_project_vote_yes_probability=0.8,  # Probability of new user voting 'Yes' on a project
    new_user_average_delegate_count=6.5,  # Average number of delegates for new users
    new_user_min_delegate_count=5,  # Minimum number of delegates for new users
    new_user_average_trustees=7.0,  # Average number of trustees for new users
    NeuronPower=xr.DataArray(),  # Empty neuron power data array
)