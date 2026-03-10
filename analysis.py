import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_elo_system(df, k_base=20, final_multiplier=0.3, semi_multiplier=0.8, prelim_multiplier=1.0):
    """
    Calculates Elo ratings for competitors over time.

    Adjusted Logic:
    - No cap on tier factor: Larger heats result in larger potential Elo swings.
    - Round Weighting: Prelims and Semis carry the most weight due to the volume of competitors.
    - Final adjustment: Finals act as the "polishing" rank adjustment.
    """
    # 1. Initialization - Calibrated to high-level trajectory
    starting_ratings = {
        'Novice': 1000,
        'Intermediate': 1050,
        'Advanced': 1200,
        'All-Star': 1350,
        'Champion': 1700
    }

    current_ratings = {}
    history = []

    # Filter out Newcomer as requested
    df = df[df['comp_level'] != 'Newcomer'].copy()

    # Strict chronological and round-based sorting
    round_order = {'prelims': 0, 'quarters': 1, 'semis': 2, 'final': 3, 'unknown': 4}
    df['round_rank'] = df['round'].map(round_order)
    df = df.sort_values(by=['cleaned_date', 'event', 'round_rank'])

    heats = df.groupby(['event', 'division', 'round', 'role'], sort=False)

    for (event, division, round_type, role), heat_df in heats:
        num_competitors = len(heat_df)
        if num_competitors < 2:
            continue

        heat_players = []
        for _, row in heat_df.iterrows():
            name = row['competitor_clean']
            level = row['comp_level']

            if name not in current_ratings:
                current_ratings[name] = starting_ratings.get(level, 1000)

            heat_players.append({
                'name': name,
                'rating': current_ratings[name],
                'rank': row['calculated_rank'],
                'level': level,
                'count': row['round_role_count']
            })

        new_changes = {p['name']: 0 for p in heat_players}

        # DYNAMIC K-FACTOR (Uncapped)
        # Removed the min(..., 1.5) cap. Now scales directly with log of competition size.
        tier_factor = np.log10(num_competitors + 1) * 1.5
        adjusted_k = k_base * tier_factor

        # Apply round-specific multipliers
        if round_type == 'final':
            adjusted_k *= final_multiplier
        elif round_type == 'semis':
            adjusted_k *= semi_multiplier
        elif round_type == 'prelims' or round_type == 'quarters':
            adjusted_k *= prelim_multiplier
        else:
            adjusted_k *= 0.5  # Default for unknown rounds

        # Multi-opponent comparison
        for i in range(len(heat_players)):
            p1 = heat_players[i]
            for j in range(len(heat_players)):
                if i == j: continue
                p2 = heat_players[j]

                expected_p1 = 1 / (1 + 10 ** ((p2['rating'] - p1['rating']) / 400))

                if p1['rank'] < p2['rank']:
                    actual_p1 = 1
                elif p1['rank'] > p2['rank']:
                    actual_p1 = 0
                else:
                    actual_p1 = 0.5

                change = adjusted_k * (actual_p1 - expected_p1)
                new_changes[p1['name']] += (change / (num_competitors - 1))

        # Finalize updates
        for p in heat_players:
            name = p['name']
            old_rating = current_ratings[name]
            new_rating = old_rating + new_changes[name]

            # Level floor protection (prevents extreme drops below division baseline)
            level_floor = starting_ratings.get(p['level'], 1000) - 50
            if new_rating < level_floor and new_changes[name] < 0:
                new_rating = old_rating + (new_changes[name] * 0.2)

            current_ratings[name] = new_rating

            history.append({
                'competitor': name,
                'date': heat_df.iloc[0]['cleaned_date'],
                'rating': new_rating,
                'level': p['level'],
                'event': event,
                'round': round_type,
                'role': role,
                'change': new_changes[name],
                'calculated_rank': p['rank'],
                'round_role_count': p['count']
            })

    return pd.DataFrame(history)


def get_competitor_history(history_df, name):
    """Returns a focused dataframe for a specific competitor's progression."""
    name_lower = name.lower()
    comp_df = history_df[history_df['competitor'].str.lower() == name_lower].copy()
    if comp_df.empty:
        return pd.DataFrame()
    return comp_df.sort_values('date')


def inspect_competitor(history_df, name):
    """Prints stats and plots the Elo history for a specific competitor."""
    comp_df = get_competitor_history(history_df, name)
    if comp_df.empty:
        print(f"No data found for competitor: {name}")
        return

    print(f"\n--- Elo Report: {name.title()} ---")
    print(f"Current Rating: {comp_df['rating'].iloc[-1]:.2f}")
    print(f"Total Heats: {len(comp_df)}")
    print(f"Highest Rating: {comp_df['rating'].max():.2f}")
    print("\nRecent History (Last 15 Heats):")
    cols = ['date', 'event', 'round', 'level', 'calculated_rank', 'round_role_count', 'rating', 'change']
    print(comp_df[cols].tail(15))

    plt.figure(figsize=(12, 6))
    plt.plot(comp_df['date'], comp_df['rating'], marker='o', label=name.title())
    plt.title(f"Elo Progression: {name.title()} (Uncapped Scaling)")
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.grid(True, alpha=0.3)
    plt.show()
    return comp_df


if __name__ == "__main__":
    try:
        path = 'input/'
        file_name = 'wcs_cleaned_2018-03_09_2026.csv'
        data = pd.read_csv(path + file_name)
        data['cleaned_date'] = pd.to_datetime(data['cleaned_date'])

        # Running with high-impact prelims and uncapped scaling
        elo_history = calculate_elo_system(data, k_base=22)

        # Inspection
        neil_data = inspect_competitor(elo_history, "neil joshi")

        # Current Leaderboard
        latest_ratings = elo_history.sort_values('date').groupby('competitor').tail(1)
        print("\n--- Current Top 10 Leaders ---")
        print(
            latest_ratings[latest_ratings['role'] == 'lead'].nlargest(10, 'rating')[['competitor', 'rating', 'level']])

    except FileNotFoundError:
        print("Please check input file path.")