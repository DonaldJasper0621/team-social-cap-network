# Team Social Capital Network

**Donald Su** — SI 507 Final Project

---

## Description

This Python command-line tool constructs and analyzes a social capital network of NBA teams for the 2016–17 season. By integrating on-court performance metrics (Offensive, Defensive, Net Ratings), economic data (player salaries, franchise values), and social media metrics (Twitter favorites/retweets), it builds a 3‑nearest-neighbor similarity graph and offers interactive exploration via a simple CLI.

## Features

- **Similarity Queries**: Find the top‑5 most similar teams to any given team.
- **Path Finding**: Compute the shortest similarity path between two teams.
- **Centrality Measures**: Rank teams by betweenness, closeness, or eigenvector centrality.
- **Team Profiles**: Display detailed stats and top‑3 scorers.
- **Filtering & Ranking**: Filter teams by any metric (e.g., `WinPct > 0.6`) and rank by fields.
- **Clustering**: Assign teams to clusters via K‑Means (k=3).
- **Visualizations**: Plot the network graph (`plot_network`) and PCA scatter (`plot_pca`).

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/team-social-cap-network.git
   cd team-social-cap-network
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install pandas numpy networkx scikit-learn matplotlib
   ```

## Data Preparation

Place the following CSV files in the project root:

- `nba_2016_2017_100.csv` (player performance stats)
- `nba_2017_players_with_salary_wiki_twitter.csv` (salary & social metrics)
- `nba_2017_att_val.csv` (franchise values)

Ensure column names match or are auto-detected by the script.

## Usage

Run the CLI:
```bash
python team_social_cap_network.py
```

At the `>>` prompt, use any of the following commands:

| Command                              | Description                                                       |
|--------------------------------------|-------------------------------------------------------------------|
| `similar <TEAM>`                     | Top‑5 most similar teams                                          |
| `path <TEAM1> <TEAM2>`               | Shortest similarity path                                          |
| `centrality <measure>`               | Rank by `betweenness`, `closeness`, or `eigenvector`            |
| `team <TEAM>`                        | Show detailed stats & top‑3 scorers                               |
| `filter <field> <op> <val>`          | Filter teams, e.g. `WinPct > 0.6`                                 |
| `rank <field>`                       | Rank teams by a field (descending)                                |
| `clusters`                           | List each team’s cluster assignment                               |
| `plot_network`                       | Visualize the similarity network graph                            |
| `plot_pca`                           | Visualize PCA projection of team features                         |
| `help`                               | Show this command menu                                            |
| `exit`                               | Quit the program                                                  |

### Example Session

```bash
>> similar GSW
SAS: distance 0.466
UTA: distance 0.512
…

>> path LAL GSW
LAL → ORL → MIA → UTA → GSW

>> centrality betweenness
MIA: 0.483
…

>> team SAS
OffRtg: 110.1, DefRtg: 102.1, NetRtg: 7.9
…

>> filter WinPct > 0.700
GSW: 0.8125
SAS: 0.7342

>> clusters
ATL: Cluster 2
BKN: Cluster 1
…

>> plot_network   # displays Figure 1
>> plot_pca       # displays Figure 2
```

## Insights & Extensions

- Elite teams cluster together by high ratings (GSW, SAS, UTA).  
- Mid‑tier powerhouses (CLE, BOS, HOU) form their own high-value cluster.  
- Rebuilding teams (LAL, BKN, SAC) occupy a distinct low-performance cluster.  
- Future work: multi-season comparisons, richer social data, interactive dashboards.

## License

This project is released under the MIT License. Feel free to fork and contribute!

