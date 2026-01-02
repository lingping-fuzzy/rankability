# CFB Team Dominance Analysis with Fixed Team IDs
#
# Author: Based on original code by Thomas R. Cameron
# Adapted to handle evolving team IDs while maintaining consistent team names
import numpy as np
from copy import deepcopy

# Define the fixed teams we want to track (from 1995)
FIXED_TEAMS = {
    1: "Clemson",
    2: "Duke",
    3: "Florida_St",
    4: "Georgia_Tech",
    5: "Maryland",
    6: "NC_State",
    7: "North_Carolina",
    8: "Virginia",
    9: "Wake_Forest"
}

abpath = 'D:/workspace/python/temp/specR-master/DataFiles/CFB/Atlantic Coast/'

def load_teams(year):
    """Load team information for a specific year and map to fixed IDs"""
    team_mapping = {}  # Maps year-specific team IDs to fixed team IDs

    try:
        with open(f'{abpath}/{year}teams.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    team_id = int(parts[0].strip())
                    team_name = parts[1].strip()

                    # Find if this team exists in our fixed list
                    for fixed_id, fixed_name in FIXED_TEAMS.items():
                        if team_name == fixed_name:
                            team_mapping[team_id] = fixed_id
                            break
    except FileNotFoundError:
        print(f"Team file for {year} not found")

    return team_mapping


def process_games(year, team_mapping):
    """Process games for a specific year and return dominance matrix"""
    # Initialize a 9x9 matrix (for our 9 fixed teams)
    dominance_matrix = np.zeros((9, 9))

    try:
        with open(f'{abpath}/{year}games.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    # Extract team IDs and scores
                    team_i_id = int(parts[2].strip())
                    score_i = int(parts[4].strip())
                    team_j_id = int(parts[5].strip())
                    score_j = int(parts[7].strip())

                    # Map to fixed team IDs
                    if team_i_id in team_mapping and team_j_id in team_mapping:
                        fixed_i = team_mapping[team_i_id] - 1  # Convert to 0-based index
                        fixed_j = team_mapping[team_j_id] - 1  # Convert to 0-based index

                        # Only process if both teams are in our fixed set
                        if score_i > score_j:
                            # Team i beat team j
                            dominance_matrix[fixed_i, fixed_j] += 1
                        elif score_i < score_j:
                            # Team j beat team i
                            dominance_matrix[fixed_j, fixed_i] += 1
                        else:
                            # Tie - add 0.5 to both directions
                            dominance_matrix[fixed_i, fixed_j] += 0.5
                            dominance_matrix[fixed_j, fixed_i] += 0.5
    except FileNotFoundError:
        print(f"Game file for {year} not found")

    return dominance_matrix


def create_dominance_list(dominance_matrix):
    """Create the ld list from a dominance matrix"""
    # Calculate net dominance
    net_dominance = dominance_matrix - dominance_matrix.T
    # Set negative values to zero
    net_dominance = np.where(net_dominance < 0, 0, net_dominance)

    # Create the dominance list (ld)
    ld = []
    for i in range(9):
        # Add self-relation (reflexive)
        ld.append((i + 1, i + 1))
        for j in range(9):
            if net_dominance[i, j] != 0 and i != j:
                ld.append((i + 1, j + 1))

    return ld


def main():
    # Process years from 1995 to 2008
    yearly_matrices = []

    for year in range(1995, 2019):
        print(f"Processing year {year}...")

        # Load team mapping for this year
        team_mapping = load_teams(year)

        # Process games and create dominance matrix
        dominance_matrix = process_games(year, team_mapping)
        print(dominance_matrix)
        yearly_matrices.append(dominance_matrix)

        # Create and print the dominance list for this year
        ld = create_dominance_list(dominance_matrix)
        print(f"Year {year} dominance list (first 10 entries): {ld[:10]}")
        print(f"Total dominance relations: {len(ld)}")
        print()

    # Create cumulative matrix across all years
    cumulative_matrix = np.sum(yearly_matrices, axis=0)

    # Create and print the cumulative dominance list
    cumulative_ld = create_dominance_list(cumulative_matrix)
    print("Cumulative dominance across all years:")
    print(cumulative_ld)

    # Print the cumulative matrix
    print("\nCumulative dominance matrix:")
    print(cumulative_matrix)


if __name__ == '__main__':
    main()