from nqg_model.logic import *
from copy import deepcopy

# Defining NQG Model Blocks
NQG_MODEL_BLOCKS: list[dict] = [
    {
        'label': 'Time Tracking',
        'ignore': False,
        'desc': 'Updates the time in the system',
        'policies': {
            'evolve_time': p_evolve_time
        },
        'variables': {
            'days_passed': s_days_passed,
            'delta_days': s_delta_days
        }
    }, 
    {
        'label': 'Onboard users',
        'policies': {},
        'variables': {
            'users': s_onboard_users
            # TODO: make sure that `users` is not mutated when doing historical run
        }
    },
    {
        'label': 'Trust & Vote',
        'policies': {
            'user_vote': p_user_vote
        },
        'variables': {
            'trustees': s_trust,
            'delegatees': replace_suf,
            'action_matrix': replace_suf,
            'vote_decision_matrix': replace_suf
            # TODO: make sure that nothing gets mutated when doing historical run
        }
    },
    {
        'label': 'Update Oracle State',
        'policies': {},
        'variables': {
            'oracle_state': s_oracle_state
            # TODO: make sure that nothing gets mutated when doing historical run
        }
    },
    {
        'label': 'Tally votes according to Neural Quorum Governance',
        'policies': {
            'tally votes': p_compute_votes
        },
        'variables': {
            'vote_power_matrix': replace_suf,
            'vote_decision_matrix': replace_suf,
            'per_project_voting': replace_suf,
            'neuron_power_tensor': replace_suf
        }
    }
]

# filter out ignored blocks
NQG_MODEL_BLOCKS = [block for block in NQG_MODEL_BLOCKS
                              if block.get('ignore', False) is False]

# Post Processing

blocks: list[dict] = []
for block in [b for b in NQG_MODEL_BLOCKS if b.get('ignore', False) != True]:
    _block = deepcopy(block)
    for variable, suf in block.get('variables', {}).items():
        if suf == add_suf:
            _block['variables'][variable] = add_suf(variable)
        elif suf == replace_suf:
            _block['variables'][variable] = replace_suf(variable)
        else:
            pass
    blocks.append(_block)


NQG_MODEL_BLOCKS = deepcopy(blocks)