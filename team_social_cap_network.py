#!/usr/bin/env python3
"""
NBA Team Social Capital Network (2016-2017 Season)

Builds a team-level network from combined on-court performance (off/def/net rating),
economic (salary), and social (Twitter) metrics. Provides a CLI for similarity queries,
shortest paths, centrality measures, filtering, clustering, and visualizations.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# --- Configuration: file paths (adjust if needed) ---
PERF_CSV = "./nba_2016_2017_100.csv"
SOC_CSV = "./nba_2017_players_with_salary_wiki_twitter.csv"
VAL_CSV = "./nba_2017_att_val.csv"


def load_and_prepare_data():
    """Load CSVs, merge player-level data, and aggregate to team-level."""
    # 1) Load performance data
    perf = pd.read_csv(PERF_CSV)
    # Normalize PLAYER and TEAM columns
    if "PLAYER" in perf.columns and "PLAYER_NAME" not in perf.columns:
        perf.rename(columns={"PLAYER": "PLAYER_NAME"}, inplace=True)
    if "TEAM" in perf.columns and "TEAM_ABBREVIATION" not in perf.columns:
        perf.rename(columns={"TEAM": "TEAM_ABBREVIATION"}, inplace=True)

    # 2) Detect and rename points column
    if "POINTS" in perf.columns:
        points_col = "POINTS"
    elif "PTS" in perf.columns:
        points_col = "PTS"
    else:
        raise KeyError("No POINTS or PTS column found in performance data")
    if points_col != "POINTS":
        perf.rename(columns={points_col: "POINTS"}, inplace=True)

    # 3) Subset performance columns
    perf_keys = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
        "W",
        "L",
        "MIN",
        "POINTS",
    ]
    perf_sub = perf[perf_keys]

    # 4) Load socio-economic + Twitter data
    soc = pd.read_csv(SOC_CSV)
    rename_map = {}
    if "PLAYER" in soc.columns and "PLAYER_NAME" not in soc.columns:
        rename_map["PLAYER"] = "PLAYER_NAME"
    if "TEAM" in soc.columns and "TEAM_ABBREVIATION" not in soc.columns:
        rename_map["TEAM"] = "TEAM_ABBREVIATION"
    # Detect salary column
    for c in soc.columns:
        if "salary" in c.lower():
            rename_map[c] = "SALARY_MILLIONS"
    # Detect any Twitter metric columns
    twitter_cols = []
    for c in soc.columns:
        lc = c.lower()
        if "twitter_follower" in lc:
            rename_map[c] = "TWITTER_FOLLOWER_COUNT"
            twitter_cols.append("TWITTER_FOLLOWER_COUNT")
        if "twitter_favorite" in lc:
            rename_map[c] = "TWITTER_FAVORITE_COUNT"
            twitter_cols.append("TWITTER_FAVORITE_COUNT")
        if "twitter_retweet" in lc:
            rename_map[c] = "TWITTER_RETWEET_COUNT"
            twitter_cols.append("TWITTER_RETWEET_COUNT")
    soc.rename(columns=rename_map, inplace=True)
    print("⏩ soc columns after rename:", soc.columns.tolist())

    soc_keys = ["PLAYER_NAME", "TEAM_ABBREVIATION", "SALARY_MILLIONS"] + twitter_cols
    soc_sub = soc[soc_keys].fillna(0)

    # 5) Merge performance and socio data
    df = perf_sub.merge(soc_sub, on=["PLAYER_NAME", "TEAM_ABBREVIATION"], how="left")
    # Fill missing salary/twitter
    df["SALARY_MILLIONS"] = df["SALARY_MILLIONS"].fillna(0)
    for tc in twitter_cols:
        df[tc] = df[tc].fillna(0)

    # 6) Consolidate social metric
    if "TWITTER_FOLLOWER_COUNT" in df.columns:
        df["SOCIAL_METRIC"] = df["TWITTER_FOLLOWER_COUNT"]
    elif twitter_cols:
        df["SOCIAL_METRIC"] = df[twitter_cols].sum(axis=1)
    else:
        df["SOCIAL_METRIC"] = 0

    # 7) Aggregate to team level
    def agg_team(g):
        return pd.Series(
            {
                "OffRtg": np.average(g["OFF_RATING"], weights=g["MIN"]),
                "DefRtg": np.average(g["DEF_RATING"], weights=g["MIN"]),
                "NetRtg": np.average(g["NET_RATING"], weights=g["MIN"]),
                "W": g["W"].max(),
                "L": g["L"].max(),
                "WinPct": g["W"].max() / (g["W"].max() + g["L"].max())
                if (g["W"].max() + g["L"].max()) > 0
                else 0,
                "TotalSalary": g["SALARY_MILLIONS"].sum(),
                "TotalSocial": g["SOCIAL_METRIC"].sum(),
            }
        )

    team_df = (
        df.groupby("TEAM_ABBREVIATION", group_keys=False).apply(agg_team).reset_index()
    )

    # 8) Load franchise valuations
    val = pd.read_csv(VAL_CSV)
    # Rename TEAM → TEAM_ABBREVIATION explicitly
    if "TEAM" in val.columns:
        val.rename(columns={"TEAM": "TEAM_ABBREVIATION"}, inplace=True)
    # Ensure the value column is named VALUE_MILLIONS
    if "VALUE_MILLIONS" not in val.columns:
        # fallback if named TOTAL
        if "TOTAL" in val.columns:
            val.rename(columns={"TOTAL": "VALUE_MILLIONS"}, inplace=True)
    print("⏩ val columns after rename:", val.columns.tolist())

    team_df = team_df.merge(
        val[["TEAM_ABBREVIATION", "VALUE_MILLIONS"]], on="TEAM_ABBREVIATION", how="left"
    ).rename(columns={"VALUE_MILLIONS": "FranchiseValue"})

    # 9) Fill any remaining NaNs in numeric columns
    numeric_cols = [
        "OffRtg",
        "DefRtg",
        "NetRtg",
        "W",
        "L",
        "WinPct",
        "TotalSalary",
        "TotalSocial",
        "FranchiseValue",
    ]
    team_df[numeric_cols] = team_df[numeric_cols].fillna(0)

    # 10) Top-3 scorers per team
    top3 = (
        df.groupby(["TEAM_ABBREVIATION", "PLAYER_NAME"])["POINTS"]
        .mean()
        .reset_index()
        .sort_values(["TEAM_ABBREVIATION", "POINTS"], ascending=[True, False])
        .groupby("TEAM_ABBREVIATION", group_keys=False)
        .head(3)
    )
    top_dict = (
        top3.groupby("TEAM_ABBREVIATION")
        .apply(
            lambda g: [f"{r.PLAYER_NAME} ({r.POINTS:.1f} PPG)" for r in g.itertuples()]
        )
        .to_dict()
    )

    return team_df, top_dict


def build_similarity_graph(team_df, k=3):
    feats = team_df[
        ["OffRtg", "DefRtg", "TotalSalary", "TotalSocial", "FranchiseValue"]
    ].values
    scaler = MinMaxScaler().fit(feats)
    norm = scaler.transform(feats)
    teams = team_df["TEAM_ABBREVIATION"].tolist()
    dist = pairwise_distances(norm, metric="euclidean")

    G = nx.Graph()
    for i, t in enumerate(teams):
        G.add_node(t, **team_df.iloc[i].to_dict())
    for i, t in enumerate(teams):
        nbrs = np.argsort(dist[i])[1 : k + 1]
        for j in nbrs:
            G.add_edge(t, teams[j], weight=float(dist[i, j]))

    return G, scaler


def compute_centralities(G):
    return {
        "betweenness": nx.betweenness_centrality(G, weight="weight", normalized=True),
        "closeness": nx.closeness_centrality(G, distance="weight"),
        "eigenvector": nx.eigenvector_centrality_numpy(G, weight="weight"),
    }


def find_similar_teams(team, team_df, scaler, k=5):
    feats = team_df[
        ["OffRtg", "DefRtg", "TotalSalary", "TotalSocial", "FranchiseValue"]
    ].values
    norm = scaler.transform(feats)
    teams = team_df["TEAM_ABBREVIATION"].tolist()
    if team not in teams:
        return []
    i = teams.index(team)
    d = np.linalg.norm(norm - norm[i], axis=1)
    idx = np.argsort(d)[1 : k + 1]
    return [(teams[j], float(d[j])) for j in idx]


def shortest_similarity_path(G, src, dst):
    try:
        return nx.shortest_path(G, source=src, target=dst, weight="weight")
    except nx.NetworkXNoPath:
        return []


def cluster_teams(team_df, scaler, method="kmeans", **kw):
    feats = team_df[
        ["OffRtg", "DefRtg", "TotalSalary", "TotalSocial", "FranchiseValue"]
    ].values
    norm = scaler.transform(feats)
    if method == "kmeans":
        model = KMeans(n_clusters=kw.get("n_clusters", 3), random_state=0)
    else:
        model = DBSCAN(eps=kw.get("eps", 0.3), min_samples=2)
    return model.fit_predict(norm)


def plot_network(G, labels=None):
    pos = nx.spring_layout(G, weight="weight", seed=42)
    plt.figure(figsize=(10, 10))
    if labels is not None:
        for lbl in set(labels):
            nodes = [n for i, n in enumerate(G.nodes()) if labels[i] == lbl]
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, node_size=300, label=f"Cluster {lbl}"
            )
    else:
        nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.legend()
    plt.title("NBA Team Similarity Network")
    plt.axis("off")
    plt.show()


def plot_pca(team_df, scaler, labels=None):
    feats = team_df[
        ["OffRtg", "DefRtg", "TotalSalary", "TotalSocial", "FranchiseValue"]
    ].values
    norm = scaler.transform(feats)
    pcs = PCA(n_components=2).fit_transform(norm)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        for lbl in set(labels):
            pts = pcs[labels == lbl]
            plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {lbl}")
    else:
        plt.scatter(pcs[:, 0], pcs[:, 1])
    for i, t in enumerate(team_df["TEAM_ABBREVIATION"]):
        plt.text(pcs[i, 0], pcs[i, 1], t, fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA of Team Features")
    plt.show()


def print_team_info(team, team_df, top_dict):
    row = team_df[team_df["TEAM_ABBREVIATION"] == team]
    if row.empty:
        print(f"Team '{team}' not found.")
        return
    r = row.iloc[0]
    print(f"\nTeam: {team}")
    print(f" OffRtg: {r.OffRtg:.1f}")
    print(f" DefRtg: {r.DefRtg:.1f}")
    print(f" NetRtg: {r.NetRtg:.1f}")
    print(f" Record: {int(r.W)}-{int(r.L)} (Win% {r.WinPct:.3f})")
    print(f" TotalSalary: ${r.TotalSalary:.2f}M")
    print(f" TotalSocial: {r.TotalSocial:.0f}")
    print(f" FranchiseValue: ${r.FranchiseValue:.0f}M")
    print(" Top Players:")
    for p in top_dict.get(team, []):
        print(f"  - {p}")
    print()


def filter_teams(team_df, field, op, val):
    ops = {
        ">": lambda x: x > val,
        "<": lambda x: x < val,
        ">=": lambda x: x >= val,
        "<=": lambda x: x <= val,
        "==": lambda x: x == val,
    }
    if field not in team_df.columns or op not in ops:
        print("Invalid filter.")
        return
    sel = team_df[ops[op](team_df[field])]
    for _, r in sel.iterrows():
        print(f"{r.TEAM_ABBREVIATION}: {r[field]}")


def rank_teams(team_df, field, asc=False):
    if field not in team_df.columns:
        print("Invalid field.")
        return
    df = team_df.sort_values(field, ascending=asc)
    for _, r in df.iterrows():
        print(f"{r.TEAM_ABBREVIATION}: {r[field]}")


def repl():
    team_df, top_dict = load_and_prepare_data()
    G, scaler = build_similarity_graph(team_df, k=3)
    centralities = compute_centralities(G)
    labels = cluster_teams(team_df, scaler, method="kmeans", n_clusters=3)

    help_text = """
Commands:
  similar <TEAM>             – top‑5 most similar teams
  path <TEAM1> <TEAM2>       – shortest similarity path
  centrality <measure>       – betweenness|closeness|eigenvector
  team <TEAM>                – show detailed info
  filter <field> <op> <val>  – e.g. WinPct > 0.6
  rank <field>               – rank teams by a field (desc)
  clusters                   – list team → cluster
  plot_network               – visualize network graph
  plot_pca                   – visualize PCA scatter
  help                       – show this menu
  exit                       – quit
"""
    print(help_text)

    while True:
        cmd = input(">> ").strip().split()
        if not cmd:
            continue
        c = cmd[0].lower()

        if c == "exit":
            break
        if c == "help":
            print(help_text)
        elif c == "similar" and len(cmd) == 2:
            for t, d in find_similar_teams(cmd[1].upper(), team_df, scaler, 5):
                print(f"{t}: distance {d:.3f}")
        elif c == "path" and len(cmd) == 3:
            path = shortest_similarity_path(G, cmd[1].upper(), cmd[2].upper())
            print(" -> ".join(path) if path else "No path found.")
        elif c == "centrality" and len(cmd) == 2 and cmd[1] in centralities:
            for t, v in sorted(centralities[cmd[1]].items(), key=lambda x: -x[1])[:10]:
                print(f"{t}: {v:.3f}")
        elif c == "team" and len(cmd) == 2:
            print_team_info(cmd[1].upper(), team_df, top_dict)
        elif c == "filter" and len(cmd) == 4:
            try:
                val = float(cmd[3])
            except ValueError:
                val = cmd[3]
            filter_teams(team_df, cmd[1], cmd[2], val)
        elif c == "rank" and len(cmd) == 2:
            rank_teams(team_df, cmd[1])
        elif c == "clusters":
            for i, t in enumerate(team_df["TEAM_ABBREVIATION"]):
                print(f"{t}: Cluster {labels[i]}")
        elif c == "plot_network":
            plot_network(G, labels)
        elif c == "plot_pca":
            plot_pca(team_df, scaler, labels)
        else:
            print("Unknown command. Type 'help'.")

    print("Goodbye!")


if __name__ == "__main__":
    repl()
