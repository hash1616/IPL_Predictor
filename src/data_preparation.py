import os
import pandas as pd
import numpy as np

# Define team name standardization mapping
TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

# Define venue standardization function
def standardize_venue(venue_name):
    if pd.isna(venue_name):
        return "Unknown Venue"
    
    # Take the part before any comma (removes city/state suffixes)
    v = str(venue_name).split(',')[0].strip()
    
    # Consolidation of variations and historical renaming
    venue_mapping = {
        'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'Feroz Shah Kotla Ground': 'Arun Jaitley Stadium',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium',
        'Punjab Cricket Association Stadium': 'Punjab Cricket Association Stadium',
        'Sardar Patel Stadium': 'Narendra Modi Stadium',
        'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
        'JSCA International Stadium Complex': 'JSCA International Cricket Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'ACA-VDCA Cricket Stadium',
        'Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'ACA-VDCA Cricket Stadium',
    }
    
    for key, val in venue_mapping.items():
        if key in v:
            v = val
            break
            
    return v

def clean_data(raw_data_path, output_dir):
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path, low_memory=False)
    
    print(f"Raw dataset loaded. Shape: {df.shape}")
    
    # 1. Clean team names in all relevant columns
    team_cols = ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_MAPPING)
            
    # 2. Standardize venue names
    df['venue'] = df['venue'].apply(standardize_venue)
    
    # 3. Clean and standardize city names
    df['city'] = df['city'].replace({'Bangalore': 'Bengaluru'})
    
    # Impute missing city names using venue name rules if any are missing or "NA"
    venue_city_map = {
        'M Chinnaswamy Stadium': 'Bengaluru',
        'Arun Jaitley Stadium': 'Delhi',
        'Wankhede Stadium': 'Mumbai',
        'Brabourne Stadium': 'Mumbai',
        'Dr DY Patil Sports Academy': 'Mumbai',
        'Eden Gardens': 'Kolkata',
        'MA Chidambaram Stadium': 'Chennai',
        'Rajiv Gandhi International Stadium': 'Hyderabad',
        'Punjab Cricket Association Stadium': 'Mohali',
        'Narendra Modi Stadium': 'Ahmedabad',
        'Maharashtra Cricket Association Stadium': 'Pune',
        'Sawai Mansingh Stadium': 'Jaipur',
    }
    
    # If city is missing or NA, fill it based on venue
    def fill_city(row):
        city = str(row['city']).strip()
        if pd.isna(row['city']) or city in ['', 'NA', 'nan', 'Unknown']:
            for venue_key, city_val in venue_city_map.items():
                if venue_key in str(row['venue']):
                    return city_val
        return row['city']
        
    df['city'] = df.apply(fill_city, axis=1)
    
    # 4. Remove duplicate records
    # Drop rows that are exact duplicates
    df = df.drop_duplicates()
    
    # 5. Filter out unmatched/invalid matches (e.g. no results)
    # Filter for matches that have a valid winner or result type is not 'no result'
    df = df[df['result_type'] != 'no result']
    df = df[df['match_won_by'].notna() & (df['match_won_by'] != 'NA') & (df['match_won_by'] != 'nan') & (df['match_won_by'] != '')]
    
    # Only keep club matches (IPL is club T20)
    if 'team_type' in df.columns:
        df = df[df['team_type'] == 'club']
        
    # Get unique match IDs
    match_ids = df['match_id'].unique()
    print(f"Number of valid unique matches after initial filtering: {len(match_ids)}")
    
    # 6. Aggregate to match-level dataset
    match_records = []
    
    for match_id in match_ids:
        match_df = df[df['match_id'] == match_id]
        
        # Check if we have innings 1 and 2 details
        innings_1 = match_df[match_df['innings'] == 1]
        if innings_1.empty:
            # If no innings 1 data exists, skip this match as it is incomplete/invalid
            continue
            
        # Extract features
        date = match_df['date'].iloc[0]
        season = match_df['season'].iloc[0]
        year = match_df['year'].iloc[0]
        venue = match_df['venue'].iloc[0]
        city = match_df['city'].iloc[0]
        toss_winner = match_df['toss_winner'].iloc[0]
        toss_decision = match_df['toss_decision'].iloc[0]
        winner = match_df['match_won_by'].iloc[0]
        
        # Team 1 is batting first, Team 2 is bowling first
        team1 = innings_1['batting_team'].iloc[0]
        team2 = innings_1['bowling_team'].iloc[0]
        
        # Validate that the winner is one of the two playing teams
        if winner not in [team1, team2]:
            # If match winner is not one of the teams playing (e.g. superover or data error), skip
            continue
            
        # Simple win margin extraction (win_outcome e.g. "140 runs", "9 wickets")
        win_outcome = match_df['win_outcome'].iloc[0]
        
        match_records.append({
            'match_id': match_id,
            'date': date,
            'season': season,
            'year': year,
            'team1': team1,
            'team2': team2,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'venue': venue,
            'city': city,
            'winner': winner,
            'win_outcome': win_outcome
        })
        
    matches_clean_df = pd.DataFrame(match_records)
    print(f"Cleaned match-level dataset shape: {matches_clean_df.shape}")
    
    # Save the match-level dataset
    os.makedirs(output_dir, exist_ok=True)
    matches_clean_path = os.path.join(output_dir, 'matches_clean.csv')
    matches_clean_df.to_csv(matches_clean_path, index=False)
    print(f"Saved cleaned matches to {matches_clean_path}")
    
    # 7. Clean and save ball-by-ball dataset (deliveries)
    # We only keep deliveries for matches that are present in our cleaned matches dataset
    clean_match_ids = set(matches_clean_df['match_id'])
    deliveries_clean_df = df[df['match_id'].isin(clean_match_ids)].copy()
    
    # Standardize team names in deliveries as well
    deliveries_clean_df['batting_team'] = deliveries_clean_df['batting_team'].replace(TEAM_MAPPING)
    deliveries_clean_df['bowling_team'] = deliveries_clean_df['bowling_team'].replace(TEAM_MAPPING)
    deliveries_clean_df['toss_winner'] = deliveries_clean_df['toss_winner'].replace(TEAM_MAPPING)
    deliveries_clean_df['match_won_by'] = deliveries_clean_df['match_won_by'].replace(TEAM_MAPPING)
    
    deliveries_clean_path = os.path.join(output_dir, 'deliveries_clean.csv')
    deliveries_clean_df.to_csv(deliveries_clean_path, index=False)
    print(f"Saved cleaned deliveries to {deliveries_clean_path}. Shape: {deliveries_clean_df.shape}")
    
    print("\n--- Data Cleaning Complete ---")

if __name__ == '__main__':
    clean_data('data/raw/IPL.csv', 'data/processed/')
