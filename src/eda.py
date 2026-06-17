import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Premium colors
PRIMARY_COLOR = '#1f77b4'  # Steel Blue
SECONDARY_COLOR = '#ff7f0e'  # Orange
SUCCESS_COLOR = '#2ca02c'  # Green
MUTED_DARK = '#2c3e50'

def run_eda(matches_path, deliveries_path, output_dir):
    print("Loading clean datasets for EDA...")
    matches_df = pd.read_csv(matches_path)
    deliveries_df = pd.read_csv(deliveries_path)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots in: {output_dir}")
    
    # ----------------------------------------------------
    # 1. Most Successful Teams (Total Wins)
    # ----------------------------------------------------
    print("1. Plotting Most Successful Teams...")
    win_counts = matches_df['winner'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=win_counts.values, y=win_counts.index, hue=win_counts.index, palette="viridis", legend=False)
    plt.title("Most Successful IPL Teams (Total Wins)", pad=15)
    plt.xlabel("Total Matches Won")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_successful_teams.png'), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # 2. Team Win Percentages
    # ----------------------------------------------------
    print("2. Plotting Team Win Percentages...")
    # Calculate matches played by each team
    matches_played = pd.concat([matches_df['team1'], matches_df['team2']]).value_counts()
    team_stats = pd.DataFrame({
        'Played': matches_played,
        'Won': win_counts
    }).fillna(0)
    team_stats['Win_Percentage'] = (team_stats['Won'] / team_stats['Played']) * 100
    team_stats = team_stats.sort_values(by='Win_Percentage', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=team_stats['Win_Percentage'], y=team_stats.index, hue=team_stats.index, palette="magma", legend=False)
    plt.title("IPL Team Win Percentages (Wins / Played)", pad=15)
    plt.xlabel("Win Percentage (%)")
    plt.ylabel("Team")
    # Annotate values on the bar
    for idx, value in enumerate(team_stats['Win_Percentage']):
        plt.text(value + 0.5, idx, f"{value:.1f}%", va='center', fontsize=9)
    plt.xlim(0, 75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'team_win_percentages.png'), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # 3. Venue Bias (First Innings Wins vs Second Innings Wins at Top Venues)
    # ----------------------------------------------------
    print("3. Plotting Venue Bias...")
    # Get top 12 venues by match count
    top_venues = matches_df['venue'].value_counts().head(12).index
    venue_df = matches_df[matches_df['venue'].isin(top_venues)].copy()
    
    # Identify if batting first or second team won
    # team1 is always batting first
    venue_df['won_batting_first'] = (venue_df['winner'] == venue_df['team1']).astype(int)
    
    venue_grouped = venue_df.groupby('venue')['won_batting_first'].agg(['count', 'sum'])
    venue_grouped['Won Batting First'] = venue_grouped['sum']
    venue_grouped['Won Batting Second'] = venue_grouped['count'] - venue_grouped['sum']
    venue_grouped = venue_grouped.drop(columns=['count', 'sum']).sort_values(by='Won Batting Second', ascending=True)
    
    ax = venue_grouped.plot(kind='barh', stacked=True, color=[MUTED_DARK, SUCCESS_COLOR], figsize=(12, 7))
    plt.title("Venue Bias: Wins Batting First vs. Batting Second (Top Venues)", pad=15)
    plt.xlabel("Number of Matches Won")
    plt.ylabel("Venue")
    plt.legend(title="Winning Side")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'venue_bias.png'), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # 4. Toss Impact (Toss Success & Decision)
    # ----------------------------------------------------
    print("4. Plotting Toss Impact...")
    # Toss winner == Match winner
    toss_match_winner_same = (matches_df['toss_winner'] == matches_df['winner']).value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Does Winning Toss Help Win the Match?
    labels = ['Toss Winner Wins Match', 'Toss Winner Loses Match']
    ax1.pie(toss_match_winner_same.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722'], explode=(0.05, 0))
    ax1.set_title("Toss Winner vs. Match Winner", pad=10)
    
    # Subplot 2: Toss Decision Distribution
    toss_decisions = matches_df['toss_decision'].value_counts()
    sns.barplot(x=toss_decisions.index, y=toss_decisions.values, hue=toss_decisions.index, ax=ax2, palette="Set2", legend=False)
    ax2.set_title("Toss Decision Distribution", pad=10)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Toss Decision")
    for idx, val in enumerate(toss_decisions.values):
        ax2.text(idx, val + 10, str(val), ha='center', fontweight='bold')
        
    plt.suptitle("Toss Impact on IPL Matches", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'toss_impact.png'), dpi=150)
    plt.close()
    
    # ----------------------------------------------------
    # 5. Year-wise Trends (Matches and Average Innings 1 Scores)
    # ----------------------------------------------------
    print("5. Plotting Year-wise Trends...")
    # Matches played per year
    matches_per_year = matches_df['year'].value_counts().sort_index()
    
    # Average first innings score per year
    # Innings 1 is the batting first innings
    first_innings = deliveries_df[deliveries_df['innings'] == 1]
    match_scores = first_innings.groupby('match_id')['runs_total'].sum().reset_index()
    match_scores_merged = pd.merge(match_scores, matches_df[['match_id', 'year']], on='match_id')
    avg_score_per_year = match_scores_merged.groupby('year')['runs_total'].mean()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Line 1: Matches Played (Left Axis)
    color = '#1f77b4'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Matches Played', color=color)
    ax1.plot(matches_per_year.index, matches_per_year.values, color=color, marker='o', linewidth=2, label='Matches Played')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(matches_per_year.index)
    ax1.tick_params(axis='x', rotation=45)
    
    # Line 2: Average 1st Innings Score (Right Axis)
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Average 1st Innings Score', color=color)
    ax2.plot(avg_score_per_year.index, avg_score_per_year.values, color=color, marker='s', linestyle='--', linewidth=2, label='Avg 1st Innings Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("IPL Matches Played and Average 1st Innings Score Trends", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'year_wise_trends.png'), dpi=150)
    plt.close()
    
    print("All plots generated and saved successfully!")

if __name__ == '__main__':
    run_eda(
        matches_path='data/processed/matches_clean.csv',
        deliveries_path='data/processed/deliveries_clean.csv',
        output_dir='assets/plots/'
    )
