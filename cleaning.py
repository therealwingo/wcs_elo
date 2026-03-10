import pandas as pd
import numpy as np
import json
import re


# Note: The data loading parts remain at the top as provided in your snippet
# df_results = pd.read_csv('input/wcs_results_2018-03_09_2026.csv')
# df_sample = df_results.sample(n=10000)
# df_sample.to_csv('input/wcs_samples_10k.csv')

def process_wcs_scores(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # 1. Basic Cleaning: Lowercase names and clean dates
    df['event'] = df['event'].str.lower()
    df['division'] = df['division'].str.lower()

    def clean_date(row):
        if pd.notna(row['event_date']) and str(row['event_date']).strip() != "":
            return pd.to_datetime(row['event_date'], errors='coerce')
        event_name = str(row['event'])
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', event_name)
        if year_match:
            return pd.to_datetime(f"{year_match.group(1)}-01-01")
        return pd.NaT

    df['cleaned_date'] = df.apply(clean_date, axis=1)

    # 2. Round Identification
    def get_round(div):
        div_l = str(div).lower()
        if 'final' in div_l: return 'final'
        if 'semi' in div_l: return 'semis'
        if 'quarter' in div_l: return 'quarters'
        if 'prelim' in div_l: return 'prelims'
        return 'unknown'

    df['round'] = df['division'].apply(get_round)

    # 3. Level Definition and Strict Filtering for Jack & Jills
    levels_map = {
        'newcomer': 'Newcomer',
        'novice': 'Novice',
        'intermediate': 'Intermediate',
        'advanced': 'Advanced',
        'all star': 'All-Star',
        'all-star': 'All-Star',
        'champion': 'Champion',
        'champ': 'Champion'
    }

    def get_comp_level(division):
        div_lower = str(division).lower()

        # EXCLUSION LIST: Remove Pro/Am, Age-based, Invitationals, and Other Styles
        exclusion_terms = [
            'pro', ' am ', '/am', 'am/', 'student', 'teacher',  # Pro/Am
            'sophisticated', 'masters', 'legends',  # Age-based
            'invitational', 'johnvitational',  # Invitationals
            'lindy', 'shag', 'hustle', 'country', 'two-step',  # Other styles
            'two step', 'ctst', 'stage', 'show', 'vintage',  # Special categories
            'retro', 'contemporary', 'criss cross', 'criss-cross'  # Specialty formats
        ]

        if any(term in div_lower for term in exclusion_terms):
            return None

        # Ensure it is a Jack and Jill
        is_jnj = any(x in div_lower for x in ['jack and jill', 'j&j', 'jack & jill', 'jack&jill'])
        if not is_jnj:
            return None

        # Check for "combo" divisions (e.g., "intermediate / advanced")
        # Exception: All-Star/Champion combos are treated as All-Star
        if '/' in div_lower or '&' in div_lower.replace('jack & jill', ''):
            found_levels = [level for level in levels_map.keys() if level in div_lower]
            unique_found = set(found_levels)

            if len(unique_found) > 1:
                # If the combo is strictly All-Star and Champion, we keep it as All-Star
                is_as_champ_combo = all(l in ['all star', 'all-star', 'champion', 'champ'] for l in unique_found)
                if is_as_champ_combo:
                    return 'All-Star'
                else:
                    return None

        # Identify WCS Level
        for key, label in levels_map.items():
            if key in div_lower:
                return label
        return None

    df['comp_level'] = df['division'].apply(get_comp_level)
    df = df.dropna(subset=['comp_level']).copy()

    # 4. Calculate Judge Scores
    def calculate_row_score(score_json):
        if pd.isna(score_json) or score_json == "": return 0.0
        try:
            scores = json.loads(score_json)
            total = 0.0
            for val in scores.values():
                val = str(val).upper()
                if val == 'Y':
                    total += 10.0
                elif val == 'A1':
                    total += 4.5
                elif val == 'A2':
                    total += 4.3
                elif val == 'A3':
                    total += 4.2
            return total
        except:
            return 0.0

    df['calculated_score'] = df['judge_scores'].apply(calculate_row_score)

    # 5. Split Competitors and Initial Role Detection
    # Name hard-coding to handle aliases/variations
    name_mapping = {
        'jeff wingo': 'jeffrey wingo'
    }

    rows = []
    for _, row in df.iterrows():
        comp_str = str(row['competitors']).lower()
        div_str = str(row['division']).lower()

        detected_role = 'unknown'
        if any(x in div_str for x in ['leader', ' lead ', ' lead']):
            detected_role = 'lead'
        elif any(x in div_str for x in ['follower', ' follow ', ' follow']):
            detected_role = 'follow'

        names = re.split(r'\s+and\s+|\s*&\s*', comp_str)
        for name in names:
            name = name.strip()
            # Apply hard-coded name mapping
            if name in name_mapping:
                name = name_mapping[name]

            if name:
                new_row = row.copy()
                new_row['competitor_clean'] = name
                new_row['role'] = detected_role
                rows.append(new_row)

    processed_df = pd.DataFrame(rows)

    # 6. Resolve Roles for Finals
    role_lookup = \
        processed_df[processed_df['role'] != 'unknown'].set_index(['competitor_clean', 'event', 'comp_level'])[
            'role'].to_dict()

    def resolve_final_role(row):
        if row['role'] == 'unknown':
            key = (row['competitor_clean'], row['event'], row['comp_level'])
            return role_lookup.get(key, 'unknown')
        return row['role']

    processed_df['role'] = processed_df.apply(resolve_final_role, axis=1)

    # 7. Count Competitors per Event-Division-Round-Role
    counts = processed_df.groupby(['event', 'division', 'round', 'role']).size().reset_index(name='round_role_count')
    processed_df = processed_df.merge(counts, on=['event', 'division', 'round', 'role'], how='left')

    # 8. Determine Final Rank
    def assign_rank(group):
        if group['round'].iloc[0] == 'final':
            group['calculated_rank'] = group['place']
        else:
            # We use method='max' with ascending=False to ensure ties for the highest
            # scores are all ranked as 1.0 (the ceiling) instead of the floor.
            group['calculated_rank'] = group['calculated_score'].rank(method='max', ascending=False)
        return group

    processed_df = processed_df.groupby(['event', 'division', 'round'], group_keys=False).apply(assign_rank)

    # 9. Apply Hard-Coded Event Dates
    event_hard_code = {
        'jjorama 2019': "2019-06-01",
        'liberty 2019': '2019-07-01'
    }

    # Overwrite cleaned_date if the event is in our dictionary
    def apply_hard_codes(row):
        event_name = row['event']
        if event_name in event_hard_code:
            return pd.to_datetime(event_hard_code[event_name])
        return row['cleaned_date']

    processed_df['cleaned_date'] = processed_df.apply(apply_hard_codes, axis=1)

    # Final sorting
    processed_df = processed_df.sort_values(by=['cleaned_date', 'event', 'division', 'calculated_rank'])

    # Save the result
    processed_df.to_csv(output_file, index=False)
    print(f"Processed file saved to: {output_file}")
    return processed_df


if __name__ == "__main__":
    input_csv = 'wcs_results_2018-03_09_2026.csv'
    output_csv = 'wcs_cleaned_2018-03_09_2026.csv'
    path = 'input/'

    try:
        df_clean = process_wcs_scores(path + input_csv, path + output_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Please ensure the file is in the same directory.")