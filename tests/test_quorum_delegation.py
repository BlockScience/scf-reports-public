from tests.default import DEFAULT_NQG_PARAMS_FOR_TESTING
from nqg_model.neural_quorum_governance import vote_from_quorum_delegation
from nqg_model.types import *
import pandas as pd
from random import choice, shuffle

import pytest as pt

from nqg_model.types import ProjectAction, Vote


def all_equal_scenario(N_user: int, N_projects: int, vote: Vote, decision: ProjectAction) -> tuple[list[str], list[str], dict[str, dict[str, ProjectAction]], dict[str, dict[str, Vote]]]:

    user_quorum: list[UserUUID] = []
    projects: list[ProjectUUID] = []
    action_matrix: ActionMatrix = {}
    decision_matrix: VoteDecisionMatrix = {}

    for i_user in range(N_user):
        user = f"user-{i_user}"
        user_quorum.append(user)
        action_matrix[user] = {}
        decision_matrix[user] = {}
        for i_proj in range(N_projects):
            project = f"project-{i_proj}"
            action_matrix[user][project] = decision
            decision_matrix[user][project] = vote
            if project not in projects:
                projects.append(project)
    return user_quorum, projects, action_matrix, decision_matrix


def random_scenario(N_user: int, N_projects: int) -> tuple[list[str], list[str], dict[str, dict[str, ProjectAction]], dict[str, dict[str, Vote]]]:

    user_quorum: list[UserUUID] = []
    projects: list[ProjectUUID] = []
    action_matrix: ActionMatrix = {}
    decision_matrix: VoteDecisionMatrix = {}

    for i_user in range(N_user):
        user = f"user-{i_user}"

        if choice([True, False]):
            user_quorum.append(user)

        action_matrix[user] = {}
        decision_matrix[user] = {}
        for i_proj in range(N_projects):
            project = f"project-{i_proj}"
            action_matrix[user][project] = choice([ProjectAction.Vote, ProjectAction.Delegate, ProjectAction.Abstain])
            decision_matrix[user][project] = choice([Vote.Yes, Vote.No, Vote.Abstain, Vote.Undefined])
            if project not in projects:
                projects.append(project)

    shuffle(user_quorum)

    return user_quorum, projects, action_matrix, decision_matrix


@pt.mark.parametrize("option", [Vote.Yes, Vote.No, Vote.Abstain])
def test_all_equal_quorum_delegation(option):
    user_quorum, projects, action_matrix, decision_matrix = all_equal_scenario(
        50, 50, option, ProjectAction.Vote)

    for project in projects:
        result = vote_from_quorum_delegation(user_quorum,
                                            project,
                                            action_matrix,
                                            decision_matrix,
                                            DEFAULT_NQG_PARAMS_FOR_TESTING)
        assert result == option


def test_repeated_quorum_delegation_over_same_random_scenario():
    user_quorum, projects, action_matrix, decision_matrix = random_scenario(50, 50)

    for project in projects:
        result_1 = vote_from_quorum_delegation(user_quorum,
                                            project,
                                            action_matrix,
                                            decision_matrix,
                                            DEFAULT_NQG_PARAMS_FOR_TESTING)

        result_2= vote_from_quorum_delegation(user_quorum,
                                            project,
                                            action_matrix,
                                            decision_matrix,
                                            DEFAULT_NQG_PARAMS_FOR_TESTING)
        
        result_3 = vote_from_quorum_delegation(user_quorum,
                                            project,
                                            action_matrix,
                                            decision_matrix,
                                            DEFAULT_NQG_PARAMS_FOR_TESTING)
        assert result_1 == result_2
        assert result_1 == result_3
