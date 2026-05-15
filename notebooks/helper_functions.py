import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import seaborn as sns  # type: ignore
import random
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from netgraph import Graph  # type: ignore
import colorsys
import networkx as nx  # type: ignore
import numpy as np
import plotly.graph_objects as go  # type: ignore
from IPython.core.display import Markdown
from IPython.display import display


def generate_users_file(users, folder_path):
    user_df = pd.DataFrame()
    # columns = _id,public_key,username
    user_df['_id'] = users['user'].unique()
    user_df['public_key'] = user_df['_id']
    user_df['username'] = user_df['_id']
    user_df.to_csv(folder_path + 'users.csv', index=False)
    return None

def generate_submissions_file(submissions, folder_path):
    submissions_df = pd.DataFrame()
    submissions_df['_id'] = submissions['project'].unique()
    submissions_df['name'] = submissions_df['_id']
    submissions_df.to_csv(folder_path + 'submissions.csv', index=False)
    return None


def projects_submitted(uniqueSubmissions):
    # Create a dataframe from the uniqueSubmissions dictionary
    df_uniqueSubmissions = pd.DataFrame({'Round': list(
        uniqueSubmissions.keys()), 'Projects Submitted': list(uniqueSubmissions.values())})

    # Sort the dataframe by the Round column
    df_uniqueSubmissions = df_uniqueSubmissions.sort_values('Round')

    # Create a new column to highlight the final bar
    df_uniqueSubmissions['Current Round'] = df_uniqueSubmissions['Round'] == max(
        df_uniqueSubmissions['Round'])

    # Create the bar chart using Plotly Express
    fig = px.bar(df_uniqueSubmissions, x='Round', y='Projects Submitted', color='Current Round',
                 color_discrete_sequence=['#5b9aa0', '#f7786b'],
                 text='Projects Submitted', title='Unique Submissions by Round (made up numbers)', )

    # Add value labels to the bars
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickmode='array', xaxis_tickvals=list(
        df_uniqueSubmissions['Round']))
    # Show the bar chart
    fig.show()
    return None


def distribution_of_votes_projects(df):
    vote_counts = df.groupby('submission')[
        'vote_type'].value_counts().unstack().fillna(0)
    vote_counts = vote_counts[['Delegate', 'No', 'Yes']].astype(int)
    vote_counts['approval_perc'] = vote_counts['Yes'] / \
        (vote_counts['Yes'] + vote_counts['No']) * 100
    vote_counts_sorted = vote_counts.sort_values(
        by='approval_perc', ascending=False)
    vote_counts_sorted['approval_perc'] = vote_counts_sorted['approval_perc'].round(
        2)  # Round to 2 decimal places
    display(Markdown(f"""### Projects by approval percentage"""))

    display(vote_counts_sorted)
    print('-'*100)
    # Creating a grouped bar chart for each submission using Plotly Express
    vote_counts_sorted['submission'] = vote_counts_sorted.index
    fig = px.bar(vote_counts_sorted, x="submission", y=["Yes", "No", "Delegate"], barmode="group",
                 color_discrete_map={"Yes": "#5b9aa0", "No": "#f7786b"},
                 title="Votes for Each Project")
    # increase the size of the plot
    fig.update_layout(width=1200, height=600)
    fig.show()

    return None


def active_users(uniqueUsers):

    # Create a dataframe from the uniqueUsers dictionary
    df_uniqueUsers = pd.DataFrame(
        {'Round': list(uniqueUsers.keys()), 'Active Users': list(uniqueUsers.values())})

    # Sort the dataframe by the Round column
    df_uniqueUsers = df_uniqueUsers.sort_values('Round')

    # Create a new column to highlight the final bar
    df_uniqueUsers['Current Round'] = df_uniqueUsers['Round'] == max(
        df_uniqueUsers['Round'])

    # Create the bar chart using Plotly Express
    fig = px.bar(df_uniqueUsers, x='Round', y='Active Users', color='Current Round',
                 color_discrete_sequence=['#5b9aa0', '#f7786b'],
                 text='Active Users', title='Active Users by Round (made up numbers)', )

    # Add value labels to the bars
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickmode='array',
                      xaxis_tickvals=list(df_uniqueUsers['Round']))
    # export to png

    # Show the bar chart
    fig.show()

    return None


def delegating_users():
    delegating_users = {}
    for round_no in range(22, 33):
        delegating_users[round_no] = delegating_users.get(
            round_no-1, 10) + random.randint(-2, 4)

    # Create a dataframe from the delegating_users dictionary
    df_delegating_users = pd.DataFrame({'Round': list(
        delegating_users.keys()), 'Delegating Users': list(delegating_users.values())})

    # Sort the dataframe by the Round column
    df_delegating_users = df_delegating_users.sort_values('Round')

    # Create a new column to highlight the final bar
    df_delegating_users['Current Round'] = df_delegating_users['Round'] == max(
        df_delegating_users['Round'])

    # Create the bar chart using Plotly Express
    fig = px.bar(df_delegating_users, x='Round', y='Delegating Users', color='Current Round',
                 color_discrete_sequence=['#5b9aa0', '#f7786b'],
                 text='Delegating Users', title='Delegating Users by Round (made up numbers)', )

    # Add value labels to the bars
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickmode='array',
                      xaxis_tickvals=list(df_delegating_users['Round']))
    # Add a trace line for the average delegating users

    # Show the bar chart
    fig.show()
    return None


def distribution_of_votes_users(df):
    df3 = df.copy()
    df3['color'] = df3['vote_type'].map(
        {'Yes': '#5b9aa0', 'No': '#f7786b', 'Delegate': '#d6d4e0'})
    # incrase the size of the plot
    fig = px.parallel_categories(df3, dimensions=[
                                 'user_ref', 'vote_type', 'submission'], color='color', title="Distribution of Votes from Voters to Projects")
    fig.update_layout(width=900, height=600)
    # add padding to left and right
    fig.update_layout(margin=dict(l=300, r=200, t=100, b=100))
    fig.show()
    print("""This visualization maps the votes from the voters (on the left) to their action (in the middle) to the projects (on the right). The mini bars on each voter and each project display the distribution of votes they cast.""")

    return None


def heatmap_of_voting_power(df):
    # Create a dataframe with submission, Trust Neuron, and Voting history columns
    df['submission'].unique()
    df_new = pd.DataFrame(
        columns=['submission', 'Trust Neuron', 'Voting history'])

    # Fill the Trust Neuron and Voting history columns with random values from 1 to 100
    df_new['Trust Neuron'] = [random.randint(
        1, 101) for i in range(len(df['submission'].unique()))]
    df_new['Voting history'] = [random.randint(
        1, 101) for i in range(len(df['submission'].unique()))]
    df_new['Expertise'] = [random.randint(
        1, 101) for i in range(len(df['submission'].unique()))]

    # Set the submission column values from the unique submissions in the original dataframe
    df_new['submission'] = df['submission'].unique()

    # Create a heatmap using plotly graph objects
    fig = go.Figure(data=go.Heatmap(
                    z=df_new[['Trust Neuron',
                              'Voting history', 'Expertise']].values,
                    x=['Trust Neuron', 'Voting history', 'Expertise'],
                    y=df_new['submission'],
                    colorscale='Greens'))

    # Customize the layout
    fig.update_layout(
        title='Neuron level Analysis',
        xaxis=dict(title='Columns', tickmode='array', tickvals=[0, 1]),
        yaxis=dict(title='Rows'),
        width=1200,  # Adjust the width of the plot
        height=1000,  # Adjust the height of the plot
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Sort by Trust Neuron",
                         method="update",
                         args=[{"z": [df_new.sort_values('Trust Neuron')[['Trust Neuron', 'Voting history', 'Expertise']].values]}]),
                    dict(label="Sort by Voting History",
                         method="update",
                         args=[{"z": [df_new.sort_values('Voting history')[['Trust Neuron', 'Voting history', 'Expertise']].values]}]),
                    dict(label="Sort by Expertise",
                         method="update",
                         args=[{"z": [df_new.sort_values('Expertise')[['Trust Neuron', 'Voting history', 'Expertise']].values]}])

                ],
                direction="right",
                showactive=True,
                x=0.5,
                y=1.05
            )
        ]
    )

    fig.show()

    print("""This heatmap shows the Trust Neuron, Voting history Neuron and Expertise Neuron's influence on each project. The values are randomly generated for demonstration purposes. 
          This Visualization can be used to identify any anomolies or excess influence of certain Neurons on the projects voting power.""")
    return None


def histogram_of_voting_power(df):
    user_list = df['user_ref'].unique()
    outliers = df['user_ref'].unique()[:3]*3  # made up numbers
    voting_power_dict = {}

    for user in user_list:
        voting_power = int(np.random.normal(20, 10))
        if voting_power < 0:
            voting_power = 0
        voting_power_dict[user] = voting_power

        # add outliers
        for user in outliers:
            voting_power_dict[user] = 75

        # Create a list of voting powers
        voting_powers = list(voting_power_dict.values())

        # Create a dataframe with the voting powers
        df_voting_powers = pd.DataFrame({'Voting Power': voting_powers})

    # Create the histogram using Plotly Express
    fig = px.histogram(df_voting_powers, x='Voting Power', nbins=20,
                       title='Distribution of Voting Power among Voters (made up numbers)')

    # Show the histogram
    fig.show()

    return None


def counterfactual_heatmap_1(counterfact_df):
    # counterfact_df.sort_values(by='backtesting', inplace=True, ascending=False)
    cmap = sns.diverging_palette(
        h_neg=20, h_pos=230, as_cmap=True, center='light', s=99)
    ax = sns.heatmap(counterfact_df, cmap=cmap,
                     annot=True, fmt='.2f', center=0)
    ax.set(xlabel='Scenarios')
    ax.set_ylabel('Projects')
    ax.set_title('Voting power received by projects').set_fontsize(12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    return ax


def counterfactual_heatmap2(counterfact_df):
    df_heatmap = counterfact_df.subtract(
        counterfact_df['1 person, 1 vote'], axis=0)
    cmap = sns.diverging_palette(
        h_neg=20, h_pos=230, as_cmap=True, center='light', s=99)
    ax = sns.heatmap(df_heatmap, cmap=cmap, annot=True, fmt='.2f', cbar_kws={
                     'label': 'Change in score compared to 1 person, 1 vote'}, center=0)
    ax.set(xlabel='Scenarios')
    ax.set_ylabel('Projects')
    ax.set_title(
        'Change in voting power compared to 1 person, 1 vote').set_fontsize(12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    return ax


def counterfactual_heatmap_3(counterfact_df):
    # counterfact_df.sort_values(by='backtesting', inplace=True, ascending=False)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    # plt.figure(figsize=(12,3), dpi=300)
    ax = sns.heatmap(counterfact_df, cmap=cmap, annot=True, fmt='.2f')
    ax.set(xlabel='Scenarios')
    ax.set_ylabel('Projects')
    ax.set_title('Projects ranked by voting power').set_fontsize(12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    return ax


def counterfactual_table(counterfact_df):
    df2 = pd.DataFrame()
    for col in counterfact_df.columns:
        ranking = counterfact_df.sort_values(by=col, ascending=False).index + ' (' + counterfact_df.sort_values(
            by=col, ascending=False)[col].apply(lambda x: f'{x:.1f}') + ')'
        df2[col] = ranking
    df2.reset_index(inplace=True)
    df2.rename_axis('Ranking', inplace=True)
    display(df2[['1 person, 1 vote', 'NQG', 'NG w/o QD', 'QD w/o NG']])


def render_trust_delegation_graphs(sim_df, i):
    t = sim_df.iloc[i].name
    trust_graph_per_day = sim_df.trustees.map(nx.DiGraph)
    delegatee_graph_per_day = sim_df.delegatees.map(nx.DiGraph)

    def sample_colors_from_hue(N):
        import colorsys
        HSV_tuples = [(x*1.0/N, 1.0, 0.8) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        return list(RGB_tuples)

    cmap = matplotlib.colormaps['BrBG']  # type: ignore

    def render_graph(G,
                     weights,
                     K=1,
                     q=1,
                     title="",
                     a=1,
                     node_color=None,
                     edge_color=None,
                     edge_alpha=0.5,
                     edge_width=0.1,
                     **kwargs):
        sizes = {k: a + K * v ** q for k, v in weights.items()}
        if node_color is None:
            node_color = {k: cmap(i / len(weights))
                          for i, k in enumerate(sorted(weights.keys()))}
        else:
            pass
        fig = plt.figure(figsize=(10, 2), dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        g = Graph(G,
                  ax=ax,
                  node_size=sizes,
                  node_edge_width=0,
                  edge_width=edge_width,
                  node_color=node_color,
                  edge_alpha=edge_alpha,
                  edge_color=edge_color,
                  **kwargs)
        ax.set_facecolor(colorsys.hsv_to_rgb(0.0, 0.0, 0.9))
        fig.set_facecolor(colorsys.hsv_to_rgb(0.0, 0.0, 0.9))  # type: ignore
        plt.title(title, fontname='Helvetica')
        plt.show()

    G = trust_graph_per_day.iloc[i]
    weights = sim_df.iloc[i].oracle_state.pagerank_results
    render_graph(G,
                 weights,
                 K=20,
                 q=0.5,  # type: ignore
                 a=0,
                 title=f"Trust Graph",
                 edge_layout='curved',
                 node_layout='spring',
                 edge_width=0.1,
                 arrows=True,
                 scale=(4, 1))

    G = delegatee_graph_per_day.iloc[i]
    weights = {k: 0 for k in G.nodes}
    weights |= {k: len(v) * 0.2 for k, v in sim_df.iloc[i].delegatees.items()}
    quorums = sim_df.iloc[i].delegatees
    source_color_map = {k: matplotlib.colormaps['tab10'](i/len(quorums))  # type: ignore
                        for i, k in enumerate(quorums.keys())}  # type: ignore
    edge_color = {(a, b): source_color_map[a] for (a, b) in G.edges}
    render_graph(G, weights,
                 K=0,
                 q=1,
                 a=2,
                 title=f'Delegation Graph',
                 edge_color=edge_color,
                 edge_width=1.0,
                 edge_alpha=0.9,
                 arrows=True,
                 edge_layout='curved',
                 node_layout='spring',
                 scale=(4, 1))


def project_voting_summary(df, round_no):
    display(Markdown(
        f"""### Total number of projects voted on in round {round_no}: **{df['submission'].nunique()}**"""))

    print('-' * 100)
    # distribution of votes among project
    distribution_of_votes_projects(df)
    print('-' * 100)
    # distribution of votes among voters
    distribution_of_votes_users(df)
    return None


def user_engagement_overview(df, round_no):
    display(Markdown(
        f"""### Total number of active users in round {round_no}: **{df['user_ref'].nunique()}**"""))
    return None


def voting_power_analysis(df):
    # heatmap of voting power
    # heatmap_of_voting_power(df)

    # print('-'*100)
    # Histogram of voting power
    # histogram_of_voting_power(df)

    return None


def trust_network_analysis(sim_df):
    render_trust_delegation_graphs(sim_df, i=1)
    return None


def get_results_counterfact_full_ranked_dfs(df: pd.DataFrame,
                                            sim_df: pd.DataFrame,
                                            scenario_map: dict[str, str],
                                            result_scenario_label='result') -> tuple[pd.DataFrame,
                                                                                   pd.DataFrame,
                                                                                   pd.DataFrame,
                                                                                   pd.DataFrame]:
    results_df = sim_df.query('timestep == 1').set_index('label')
    counterfact_df = results_df['per_project_voting'].apply(
        pd.Series).transpose()
    counterfact_df.sort_values(by='backtesting', inplace=True, ascending=False)
    counterfact_df.rename(columns=scenario_map, inplace=True)
    counterfact_df = counterfact_df[scenario_map.values()]

    full_df = counterfact_df.copy()
    full_df[result_scenario_label] = df.groupby('submission').tally_vote_power.sum()
    ranked_full_df = full_df.copy().rank(ascending=False)
    return (results_df, counterfact_df, full_df, ranked_full_df)
