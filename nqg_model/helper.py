from nqg_model.types import *
import numpy as np
import pandas as pd

def retrieve_prev_state_users(history: list[list[NQGModelState]]) -> set[UserUUID]:
    """
    Calculate the past voting learning curve weights.

    This function generates a mapping of round indices to their corresponding weights 
    based on an exponential learning curve.

    Parameters:
        first_round_scored (int): The first round to be scored. Default is 21.
        current_round (int): The current round. Default is 26.
        W_max (float): The maximum weight. Default is 1.
        lambda_ (float): The decay constant for the exponential function. Default is 0.5.

    Returns:
        dict[int, float]: A dictionary mapping round indices to their corresponding weights.
    """
    if len(history) > 1:
        previous_state_users = set(u.label 
                                for u 
                                in history[-1][-1]['users'])
    else:
        previous_state_users = set()
    return previous_state_users


def generate_users_file(users, folder_path):
    """
    Generate a CSV file containing user information.

    This function creates a CSV file with user IDs, public keys, and usernames.

    Parameters:
        users (pd.DataFrame): A DataFrame containing user information.
        folder_path (str): The folder path where the CSV file will be saved.

    Returns:
        None
    """
    user_df = pd.DataFrame()
    # columns = _id,public_key,username
    user_df['_id'] = users['user'].unique()
    user_df['public_key'] = user_df['_id']
    user_df['username'] = user_df['_id']
    user_df.to_csv(folder_path + 'users.csv', index=False)
    return None

def generate_submissions_file(submissions, folder_path):
    """
    Generate a CSV file containing submission information.

    This function creates a CSV file with submission IDs and names.

    Parameters:
        submissions (pd.DataFrame): A DataFrame containing submission information.
        folder_path (str): The folder path where the CSV file will be saved.

    Returns:
        None
    """
    submissions_df = pd.DataFrame()
    submissions_df['_id'] = submissions['project'].unique()
    submissions_df['name'] = submissions_df['_id']
    submissions_df.to_csv(folder_path + 'submissions.csv', index=False)
    return None
