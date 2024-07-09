from typing import Annotated, TypedDict, Callable, Set, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto, Flag
import xarray as xr


# Type Aliases for clarity
Days = Annotated[float, 'days']  # Number of days
UserUUID = str  # Unique identifier for a user
ProjectUUID = str  # Unique identifier for a project
VotingPower = float  # The voting power a user has
PastRoundIndex = int  # Index representing past rounds

class CounterfactualFlags(Flag):
    """
    Enumeration for different counterfactual scenarios.
    """
    none = 0
    replace_delegate_by_abstain = auto()
    replace_neuron_governance_by_one_vote_one_power = auto()
    remove_first_neuron_layer = auto()
    remove_second_neuron_layer = auto()
    add_bug_in_neuron_governance = auto()

class ReputationCategory(Enum):
    """
    Enumeration for different user reputation categories.
    """
    Pilot = auto()
    Navigator = auto()
    Pathfinder = auto()
    Verified = auto()
    Unknown = auto()

# Graph types
TrustGraph = dict[UserUUID, set[UserUUID]]  # Trust relationships between users
DelegationGraph = dict[UserUUID, list[UserUUID]]  # Delegation relationships between users
UserReputations = dict[UserUUID, ReputationCategory]  # User reputations
UserPriorVoting = dict[UserUUID, list[int]]  # User prior voting history



@dataclass
class OracleState:
    """
    Data class to hold the state of the Oracle.
    """
    pagerank_results: dict[UserUUID, float]  # PageRank results for users
    reputation_bonus_values: UserReputations  # Reputation bonuses for users
    prior_voting_bonus_values: UserPriorVoting  # Prior voting bonuses for users
    reputation_bonus_map: dict[ReputationCategory, float]  # Mapping of reputation categories to bonus values
    prior_voting_bonus_map: dict[int, float]  # Mapping of past rounds to bonus values

class Vote(float, Enum):
    """
    The Voting Actions towards a Project that a User can take and the 
    values in terms of Voting Power.
    """
    Yes = 1.0
    No = -1.0
    Abstain = 0.0
    Undefined = float('nan')

class ProjectAction(Enum):
    """
    The Decisions that a User can make in regards to a Project.
    """
    Vote = auto()
    Delegate = auto()
    Abstain = auto()

@dataclass
class User:
    """
    Data class to hold information about a User.
    """
    label: UserUUID  # Unique identifier for the user
    reputation: ReputationCategory  # User's reputation category
    active_past_rounds: Set[PastRoundIndex]  # Set of past rounds the user was active in

# Action and Vote matrices
ActionMatrix = dict[UserUUID, dict[ProjectUUID, ProjectAction]]  # Actions of users on projects
VoteDecisionMatrix = dict[UserUUID, dict[ProjectUUID, Vote]]  # Voting decisions of users on projects
VotePowerMatrix = dict[UserUUID, dict[ProjectUUID, VotingPower]]  # Voting power of users on projects
PerProjectVoting = dict[ProjectUUID, VotingPower]  # Total voting power per project

# Function types for Oracle and Neuron computations
OracleFunction = Callable[[UserUUID, ProjectUUID, VotingPower, OracleState], VotingPower]
WeightingFunction = Callable[[VotingPower], VotingPower]
LayerAggregatorFunction = Callable[[list[VotingPower]], VotingPower]
Neuron = tuple[OracleFunction, WeightingFunction]  # Tuple representing a Neuron
NeuronsContainer = dict[Annotated[str, 'Neuron label'], Neuron]  # Container for Neurons
NeuronLayer = tuple[NeuronsContainer, LayerAggregatorFunction]  # Layer of Neurons

# Defining Neuron Power as an xarray DataArray
NeuronPower = xr.DataArray




class NQGModelState(TypedDict):
    """
    Typed dictionary representing the state of the NQG Model.
    """
    days_passed: Days
    delta_days: Days
    users: list[User]
    
    delegatees: DelegationGraph
    trustees: TrustGraph
    
    action_matrix: ActionMatrix
    vote_decision_matrix: VoteDecisionMatrix
    vote_power_matrix: VotePowerMatrix
    per_project_voting: PerProjectVoting
    oracle_state: OracleState
    neuron_power_tensor: NeuronPower

class NQGModelParams(TypedDict):
    """
    Typed dictionary representing the parameters of the NQG Model.
    """
    label: str
    timestep_in_days: Days
    counterfactual_flags: CounterfactualFlags

    # Quorum Delegation Parameters
    quorum_agreement_weight_yes: float
    quorum_agreement_weight_no: float
    quorum_agreement_weight_abstain: float
    max_quorum_selected_delegates: int
    max_quorum_candidate_delegates: int
    quorum_delegation_absolute_threshold: float
    quorum_delegation_relative_threshold: float

    # Neural Governance Parameters
    neuron_layers: list[NeuronLayer]
    initial_power: float
    NeuronPower: NeuronPower

    # Neuron parameters
    past_rounds: Set[PastRoundIndex]

    # Exogenous parameters
    projects: Set[ProjectUUID]

    # Behavioral Parameters
    avg_new_users_per_day: float
    avg_user_past_votes: float

    new_user_action_probability: float
    
    new_user_project_vote_probability: float
    new_user_project_vote_yes_probability: float
    
    new_user_average_delegate_count: float
    new_user_min_delegate_count: int
    new_user_average_trustees: float

class TimestepSignal(TypedDict):
    """
    Typed dictionary for timestep signals.
    """
    delta_days: float

class UserActionSignal(TypedDict):
    """
    Typed dictionary for user action signals.
    """
    delegatees: DelegationGraph
    action_matrix: ActionMatrix
    vote_decision_matrix: VoteDecisionMatrix

class IDMaps(NamedTuple):
    """
    Named tuple to map various IDs to their corresponding entities.
    """
    UserID2Wallet: dict  # Mapping of user IDs to wallet addresses
    UserID2Handle: dict  # Mapping of user IDs to handles
    SubmissionID2Label: dict  # Mapping of submission IDs to labels
