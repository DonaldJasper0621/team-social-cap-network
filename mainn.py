import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# ---------------------------
# 1. Load and Preview the Data
# ---------------------------
# Adjust the filename/path as needed.
data_file = 'nba_2016_2017.csv'
df = pd.read_csv(data_file)

# Display the first few rows to show that data has been "touched"
print("Data preview:")
print(df.head())

# ---------------------------
# 2. Aggregate Player Metrics to Team-Level
# ---------------------------
# For the team-level profile, we will aggregate key performance and social metrics.
# You can choose the aggregation method (mean, sum, etc.) as appropriate.
# In this example, we compute the average of the following columns for each team:
# - OFF_RATING, DEF_RATING, NET_RATING
# - W_PCT (win percentage)
# - TWITTER_FOLLOWER_COUNT_MILLIONS (social metric)
# - SALARY_MILLIONS (economic metric)

# First, select columns we want to aggregate. Adjust if needed.
cols_to_aggregate = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'W_PCT', 
                       'TWITTER_FOLLOWER_COUNT_MILLIONS', 'SALARY_MILLIONS']

# Group by team abbreviation (you can also group by TEAM_ID if preferred)
team_stats = df.groupby('TEAM_ABBREVIATION')[cols_to_aggregate].mean().reset_index()

print("\nTeam Aggregated Stats:")
print(team_stats)

# ---------------------------
# 3. Normalize the Aggregated Data and Compute Similarity
# ---------------------------
# We'll normalize the aggregated features to ensure all metrics are on a comparable scale.
scaler = StandardScaler()
features = team_stats[cols_to_aggregate]
features_scaled = scaler.fit_transform(features)

# Add the scaled features back to our dataframe for later reference (optional)
scaled_df = pd.DataFrame(features_scaled, columns=[f"{col}_scaled" for col in cols_to_aggregate])
team_stats_scaled = pd.concat([team_stats[['TEAM_ABBREVIATION']], scaled_df], axis=1)
print("\nTeam Aggregated Stats (Scaled):")
print(team_stats_scaled)

# Build a similarity matrix based on Euclidean distance between team profiles.
# For this example, we compute pairwise Euclidean distances.
teams = team_stats_scaled['TEAM_ABBREVIATION'].tolist()
n_teams = len(teams)

# Create an empty similarity (distance) dictionary
similarity = {}

for i in range(n_teams):
    team_i = teams[i]
    vec_i = features_scaled[i]
    similarity[team_i] = {}
    for j in range(n_teams):
        team_j = teams[j]
        vec_j = features_scaled[j]
        # Euclidean distance as a measure of dissimilarity
        dist = euclidean(vec_i, vec_j)
        similarity[team_i][team_j] = dist

# ---------------------------
# 4. Build the Network Graph
# ---------------------------
# Each node represents a team. An edge is added between every pair with a weight equal to the similarity distance.
# You could choose to add only edges below a certain distance threshold to sparsify the network.
G = nx.Graph()

# Add nodes
for team in teams:
    G.add_node(team)

# Add edges: Here we create a complete graph weighted by the distance.
for i in range(n_teams):
    for j in range(i+1, n_teams):
        team_i = teams[i]
        team_j = teams[j]
        dist = similarity[team_i][team_j]
        G.add_edge(team_i, team_j, weight=dist)

# Optionally, draw the graph to visualize (nodes positioned by spring layout)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=12)
plt.title("Team Social Capital Network")
plt.axis('off')
plt.show()

# ---------------------------
# 5. User Interaction Functions
# ---------------------------
def find_similar_teams(team, top_n=3):
    """
    Given a team name, find the top_n most similar teams based on the aggregated metrics.
    Similarity is based on the smallest Euclidean distance.
    """
    if team not in similarity:
        print(f"Team '{team}' not found.")
        return
    
    # Exclude the team itself, then sort by distance.
    sorted_teams = sorted(similarity[team].items(), key=lambda x: x[1])
    similar = [t for t in sorted_teams if t[0] != team][:top_n]
    print(f"\nTop {top_n} teams similar to {team}:")
    for other_team, dist in similar:
        print(f"{other_team} with distance {dist:.3f}")

def shortest_path_between(team_a, team_b):
    """
    Finds and prints the shortest path between two teams using edge weights.
    """
    if team_a not in G.nodes() or team_b not in G.nodes():
        print("One or both teams not found in the network.")
        return
    try:
        path = nx.shortest_path(G, source=team_a, target=team_b, weight='weight')
        path_length = nx.shortest_path_length(G, source=team_a, target=team_b, weight='weight')
        print(f"\nShortest path between {team_a} and {team_b}:")
        print(" -> ".join(path))
        print(f"Total path distance: {path_length:.3f}")
    except nx.NetworkXNoPath:
        print(f"No path found between {team_a} and {team_b}.")

def most_connected_team():
    """
    Determines and prints the team that is the most connected in the network,
    based on degree centrality.
    """
    centrality = nx.degree_centrality(G)
    most_connected = max(centrality.items(), key=lambda x: x[1])
    print(f"\nThe most connected team is {most_connected[0]} with a centrality of {most_connected[1]:.3f}")

def get_team_stat(team):
    """
    Provides the aggregated statistics for a given team.
    """
    team_data = team_stats[team_stats['TEAM_ABBREVIATION'] == team]
    if team_data.empty:
        print(f"Team '{team}' not found.")
    else:
        print(f"\nAggregated stats for {team}:")
        print(team_data.to_string(index=False))

# ---------------------------
# 6. Example Command Line Interface
# ---------------------------
def main():
    while True:
        print("\nSelect an interaction:")
        print("1: Find most similar teams")
        print("2: Find shortest path between two teams")
        print("3: Find the most connected team")
        print("4: Get aggregated stats for a team")
        print("5: Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        if choice == '1':
            team = input("Enter the team abbreviation (e.g., LAL for Los Angeles Lakers): ").strip().upper()
            find_similar_teams(team)
        elif choice == '2':
            team_a = input("Enter the source team abbreviation: ").strip().upper()
            team_b = input("Enter the target team abbreviation: ").strip().upper()
            shortest_path_between(team_a, team_b)
        elif choice == '3':
            most_connected_team()
        elif choice == '4':
            team = input("Enter the team abbreviation: ").strip().upper()
            get_team_stat(team)
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
