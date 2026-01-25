import streamlit as st
import pandas as pd
import plotly.express as px
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import numpy as np
import plotly.graph_objects as go 

# PAGE CONFIG & AUTH
st.set_page_config(
    page_title="CBB Analytics",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.get('selection') == "🏠 Home" else "collapsed"
)

def get_private_key():
    p_key_text = st.secrets["connections"]["snowflake"]["private_key_content"]
    passphrase = st.secrets["connections"]["snowflake"].get("private_key_passphrase")
    p_key_obj = serialization.load_pem_private_key(
        p_key_text.encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend()
    )
    return p_key_obj.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

# DATA CACHING
@st.cache_data(ttl=3600)
def load_all_data():
    conn = st.connection("snowflake", private_key=get_private_key())
    df_teams = conn.query("SELECT * FROM cbb_data.views.dim_team")
    df_ato = conn.query("SELECT * FROM cbb_data.views.fact_ato_results")
    df_games = conn.query("SELECT * FROM cbb_data.views.fact_game_team_stats")
    df_crushers = conn.query("SELECT * FROM cbb_data.views.fact_consecutive_possessions_offensive_crushers")
    df_kills = conn.query("SELECT * FROM cbb_data.views.fact_consecutive_possessions_defensive_kills")
    df_scoring_runs = conn.query("SELECT * FROM cbb_data.views.fact_scoring_runs")
    df_lineup_stats = conn.query("select * from cbb_data.views.fact_lineup_stats")
    return df_teams, df_ato, df_games, df_crushers, df_kills, df_scoring_runs, df_lineup_stats
df_teams, df_ato, df_games, df_crushers, df_kills, df_scoring_runs, df_lineup_stats = load_all_data()

team_list = sorted(df_teams['TEAM_NAME'].unique())
try:
    default_ix = team_list.index("Duke")
except ValueError:
    default_ix = 0

# COSMETIC FUNCTIONS 
def draw_card(label, val, p_glob, p_conf, p_tier, conf_label):
    st.markdown(f"""
        <div style="border: 1px solid #444; border-radius: 10px; padding: 12px; text-align: center; background-color: #1e1e1e; min-height: 160px;">
            <div style="color: #bbb; font-size: 0.85rem;">{label}</div>
            <div style="font-size: 1.6rem; font-weight: bold; margin: 5px 0;">{val}</div>
            <div style="background-color: {get_p_color(p_glob)}; border-radius: 3px; font-size: 0.7rem; margin: 2px 0; padding: 2px;">NCAA: {get_ordinal(p_glob)}</div>
            <div style="background-color: {get_p_color(p_conf)}; border-radius: 3px; font-size: 0.7rem; margin: 2px 0; padding: 2px;">{conf_label}: {get_ordinal(p_conf)}</div>
            <div style="background-color: {get_p_color(p_tier)}; border-radius: 3px; font-size: 0.7rem; margin: 2px 0; padding: 2px;">Tier: {get_ordinal(p_tier)}</div>
        </div>
    """, unsafe_allow_html=True)

def get_ordinal(n):
    n = int(n)
    if 11 <= (n % 100) <= 13: return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"

def get_p_color(p):
    if p >= 80: return "rgba(40, 167, 69, 0.6)" # Green
    if p >= 50: return "rgba(253, 126, 20, 0.6)" # Orange
    return "rgba(220, 53, 69, 0.6)"             # Red

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("Reports")
    selection = st.radio("Select an Analytics Report", 
            ["🏠 Home", 
             "📊 Team Breakdown", 
             "⏱️ After Timeout Efficiency",
             "🔥 Momentum & Adjustments"
             ],
        key="main_nav"
    )
    
    st.divider()
    selected_team = st.selectbox("Selected Team Focus", team_list, index=default_ix)

# REPORT DISPLAYS

# * HOME PAGE *
if selection == "🏠 Home":
    st.title("College Basketball - Advanced Analytics")
    st.header("Welcome to my sports analytics website focusing on college basketball.")
    st.markdown("This website is designed to tackle situational analytics," \
    " helping look at the game through a different lens. " \
    "Select a report on the left to begin.")

# * TEAM BREAKDOWN *
elif selection == "📊 Team Breakdown":
    # --- 1. DATA PREP & GLOBAL FILTER ---
    df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
    raw_team_games = df_games[df_games['TEAM_NAME'] == selected_team].copy()
    
    st.markdown(f"## :bar_chart: {selected_team} Performance Breakdown")
    
    t1, t2 = st.columns([1, 2])
    with t1:
        power_filter = st.toggle("⚡ Power Conference Opponents Only", value=False)
    
    if power_filter:
        team_games = raw_team_games[raw_team_games['OPPONENT_CONFERENCE_TYPE'] == 'Power'].copy()
        filter_status = " (vs. Power Conf)"
    else:
        team_games = raw_team_games.copy()
        filter_status = ""

    if team_games.empty:
        st.warning(f"No games found for {selected_team} against Power Conference opponents.")
        st.stop()

    # Capture valid Game IDs for consistent filtering in the Runs section
    valid_game_ids = team_games['GAME_ID'].tolist()

    team_games = team_games.sort_values('GAME_DATE', ascending=True)
    team_games['Wins_Cum'] = team_games['TEAM_VICTORY_INDICATOR'].cumsum()
    team_games['Losses_Cum'] = (~team_games['TEAM_VICTORY_INDICATOR']).cumsum()
    team_games['Record_Str'] = team_games.apply(
        lambda x: f"{'W' if x['TEAM_VICTORY_INDICATOR'] else 'L'} ({int(x['Wins_Cum'])}-{int(x['Losses_Cum'])})", axis=1
    )

    # --- 2. COLLAPSIBLE GAME LOG (NEW POSITION) ---
    with st.expander(f":calendar: View Full Game Log{filter_status}", expanded=False):
        log_final = team_games.sort_values('GAME_DATE', ascending=False)[['GAME_DATE', 'OPPONENT_TEAM_NAME', 'Record_Str', 'TEAM_POINTS', 'OPPONENT_POINTS']].copy()
        log_final['GAME_DATE'] = log_final['GAME_DATE'].dt.date
        # Formatting scores back to rounded integers
        log_final['TEAM_POINTS'] = log_final['TEAM_POINTS'].fillna(0).astype(int)
        log_final['OPPONENT_POINTS'] = log_final['OPPONENT_POINTS'].fillna(0).astype(int)
        log_final.columns = ['Date', 'Opponent', 'Result', 'Score', 'Opp Score']
        
        def style_row(row):
            color = 'background-color: rgba(0,255,0,0.05)' if 'W' in str(row.Result) else 'background-color: rgba(255,0,0,0.05)'
            return [color]*len(row)
        
        st.dataframe(log_final.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)

    # --- 3. PERCENTILE ENGINE & TRUE % UTILS ---
    team_info = df_teams[df_teams['TEAM_NAME'] == selected_team].iloc[0]
    conf, tier = team_info['CONFERENCE'], team_info['CONFERENCE_TYPE']
    team_blue = f"<span style='color: #3B12F5; text-decoration: underline; font-weight: bold;'>{selected_team}</span>"

    # Define the raw columns needed for true calculation (Updated with your _ATTEMPT naming)
    raw_stat_cols = [
        'TEAM_POINTS', 'TEAM_2PT_FG_MADE', 'TEAM_2PT_FG_ATTEMPT', 'TEAM_3PT_FG_MADE', 'TEAM_3PT_FG_ATTEMPT', 'TEAM_FT_MADE', 'TEAM_FT_ATTEMPT',
        'OPPONENT_POINTS', 'OPPONENT_2PT_FG_MADE', 'OPPONENT_2PT_FG_ATTEMPT', 'OPPONENT_3PT_FG_MADE', 'OPPONENT_3PT_FG_ATTEMPT', 'OPPONENT_FT_MADE', 'OPPONENT_FT_ATTEMPT'
    ]
    
    # Aggregating for all teams
    league_totals = df_games.groupby(['TEAM_NAME', 'CONFERENCE', 'CONFERENCE_TYPE'])[raw_stat_cols].sum().reset_index()

    # FIX: Explicitly force numeric/float to prevent Decimal math errors
    for col in raw_stat_cols:
        league_totals[col] = pd.to_numeric(league_totals[col], errors='coerce').fillna(0).astype(float)
        team_games[col] = pd.to_numeric(team_games[col], errors='coerce').fillna(0).astype(float)

    # Helper to calculate percentages for a dataframe
    def add_true_percentages(df):
        # Offensive
        df['TEAM_2PT_FG_PERCENT'] = df['TEAM_2PT_FG_MADE'] / df['TEAM_2PT_FG_ATTEMPT']
        df['TEAM_3PT_FG_PERCENT'] = df['TEAM_3PT_FG_MADE'] / df['TEAM_3PT_FG_ATTEMPT']
        df['TEAM_FT_PERCENT'] = df['TEAM_FT_MADE'] / df['TEAM_FT_ATTEMPT']
        df['TEAM_EFFECTIVE_FG_PERCENT'] = ((df['TEAM_2PT_FG_MADE'] + df['TEAM_3PT_FG_MADE']) + (0.5 * df['TEAM_3PT_FG_MADE'])) / (df['TEAM_2PT_FG_ATTEMPT'] + df['TEAM_3PT_FG_ATTEMPT'])
        
        # Defensive
        df['OPPONENT_2PT_FG_PERCENT'] = df['OPPONENT_2PT_FG_MADE'] / df['OPPONENT_2PT_FG_ATTEMPT']
        df['OPPONENT_3PT_FG_PERCENT'] = df['OPPONENT_3PT_FG_MADE'] / df['OPPONENT_3PT_FG_ATTEMPT']
        df['OPPONENT_FT_PERCENT'] = df['OPPONENT_FT_MADE'] / df['OPPONENT_FT_ATTEMPT']
        df['OPPONENT_EFFECTIVE_FG_PERCENT'] = ((df['OPPONENT_2PT_FG_MADE'] + df['OPPONENT_3PT_FG_MADE']) + (0.5 * df['OPPONENT_3PT_FG_MADE'])) / (df['OPPONENT_2PT_FG_ATTEMPT'] + df['OPPONENT_3PT_FG_ATTEMPT'])
        
        return df

    all_teams_bench = add_true_percentages(league_totals)
    league_ppg = df_games.groupby(['TEAM_NAME'])[['TEAM_POINTS', 'OPPONENT_POINTS']].mean().reset_index()
    all_teams_bench = all_teams_bench.drop(columns=['TEAM_POINTS', 'OPPONENT_POINTS']).merge(league_ppg, on='TEAM_NAME')

    def get_pct(col, val, group_df=all_teams_bench):
        if group_df.empty: return 0
        dist = group_df[col].dropna()
        return (dist > val).mean() * 100 if "OPPONENT" in col else (dist < val).mean() * 100

    # --- 4. OFFENSIVE & DEFENSIVE PROFILES ---
    st.markdown(f"### Performance Profile {filter_status}", unsafe_allow_html=True)
    team_totals = team_games[raw_stat_cols].sum()
    
    # Offensive Section
    st.subheader(":basketball: Offensive Profile")
    off_profiles = [
        ("Points", team_games['TEAM_POINTS'].mean(), "TEAM_POINTS", ".1f"),
        ("2PT %", team_totals['TEAM_2PT_FG_MADE'] / team_totals['TEAM_2PT_FG_ATTEMPT'] if team_totals['TEAM_2PT_FG_ATTEMPT'] > 0 else 0, "TEAM_2PT_FG_PERCENT", ".1%"),
        ("3PT %", team_totals['TEAM_3PT_FG_MADE'] / team_totals['TEAM_3PT_FG_ATTEMPT'] if team_totals['TEAM_3PT_FG_ATTEMPT'] > 0 else 0, "TEAM_3PT_FG_PERCENT", ".1%"),
        ("FT %", team_totals['TEAM_FT_MADE'] / team_totals['TEAM_FT_ATTEMPT'] if team_totals['TEAM_FT_ATTEMPT'] > 0 else 0, "TEAM_FT_PERCENT", ".1%"),
        ("eFG %", ((team_totals['TEAM_2PT_FG_MADE'] + team_totals['TEAM_3PT_FG_MADE']) + (0.5 * team_totals['TEAM_3PT_FG_MADE'])) / (team_totals['TEAM_2PT_FG_ATTEMPT'] + team_totals['TEAM_3PT_FG_ATTEMPT']) if (team_totals['TEAM_2PT_FG_ATTEMPT'] + team_totals['TEAM_3PT_FG_ATTEMPT']) > 0 else 0, "TEAM_EFFECTIVE_FG_PERCENT", ".1%")
    ]
    
    c_off = st.columns(5)
    for i, (lab, val, col, fmt) in enumerate(off_profiles):
        p_g = get_pct(col, val)
        p_c = get_pct(col, val, all_teams_bench[all_teams_bench['CONFERENCE'] == conf])
        p_t = get_pct(col, val, all_teams_bench[all_teams_bench['CONFERENCE_TYPE'] == tier])
        with c_off[i]: draw_card(lab, f"{val:{fmt}}", p_g, p_c, p_t, conf)

    # Defensive Section
    st.write("")
    st.subheader(":shield: Defensive Profile")
    def_profiles = [
        ("Pts Allowed", team_games['OPPONENT_POINTS'].mean(), "OPPONENT_POINTS", ".1f"),
        ("Opp 2PT %", team_totals['OPPONENT_2PT_FG_MADE'] / team_totals['OPPONENT_2PT_FG_ATTEMPT'] if team_totals['OPPONENT_2PT_FG_ATTEMPT'] > 0 else 0, "OPPONENT_2PT_FG_PERCENT", ".1%"),
        ("Opp 3PT %", team_totals['OPPONENT_3PT_FG_MADE'] / team_totals['OPPONENT_3PT_FG_ATTEMPT'] if team_totals['OPPONENT_3PT_FG_ATTEMPT'] > 0 else 0, "OPPONENT_3PT_FG_PERCENT", ".1%"),
        ("Opp FT %", team_totals['OPPONENT_FT_MADE'] / team_totals['OPPONENT_FT_ATTEMPT'] if team_totals['OPPONENT_FT_ATTEMPT'] > 0 else 0, "OPPONENT_FT_PERCENT", ".1%"),
        ("Opp eFG %", ((team_totals['OPPONENT_2PT_FG_MADE'] + team_totals['OPPONENT_3PT_FG_MADE']) + (0.5 * team_totals['OPPONENT_3PT_FG_MADE'])) / (team_totals['OPPONENT_2PT_FG_ATTEMPT'] + team_totals['OPPONENT_3PT_FG_ATTEMPT']) if (team_totals['OPPONENT_2PT_FG_ATTEMPT'] + team_totals['OPPONENT_3PT_FG_ATTEMPT']) > 0 else 0, "OPPONENT_EFFECTIVE_FG_PERCENT", ".1%")
    ]
    
    c_def = st.columns(5)
    for i, (lab, val, col, fmt) in enumerate(def_profiles):
        p_g = get_pct(col, val)
        p_c = get_pct(col, val, all_teams_bench[all_teams_bench['CONFERENCE'] == conf])
        p_t = get_pct(col, val, all_teams_bench[all_teams_bench['CONFERENCE_TYPE'] == tier])
        with c_def[i]: draw_card(lab, f"{val:{fmt}}", p_g, p_c, p_t, conf)

# --- 4.5 LINEUP EFFICIENCY (TOP 5 / BOTTOM 5) ---
    st.divider()
    
    # 1. Configuration UI
    st.markdown(f"### 🖐️ Lineup Efficiency: {team_blue}{filter_status}", unsafe_allow_html=True)
    
    # Slider for sample size control
    min_mins_filter = st.slider("Minimum Minutes Played Threshold", 1, 30, 5)

    # 2. Filter and Prep Data
    team_lineups = df_lineup_stats[df_lineup_stats['TEAM_NAME'] == selected_team].copy()
    
    if team_lineups.empty:
        st.info("No lineup data available for this team.")
    else:
        team_lineups['minutes'] = team_lineups['TOTAL_SECONDS'] / 60
        max_mins = team_lineups['minutes'].max()

        # Filter by the slider value
        qualified_lineups = team_lineups[team_lineups['minutes'] >= min_mins_filter].copy()
        
        if qualified_lineups.empty:
            st.warning(f"No lineups have played more than {min_mins_filter} minutes. Try lowering the threshold.")
        else:
            top_5 = qualified_lineups.sort_values('NET_RATING', ascending=False).head(5)
            bot_5 = qualified_lineups.sort_values('NET_RATING', ascending=True).head(5)

            col_top, col_bot = st.columns(2)

            with col_top:
                st.subheader("✅ Top 5 Lineups")
                for _, row in top_5.iterrows():
                    with st.container(border=True):
                        # Header: Rating and Minutes
                        c1, c2 = st.columns([1, 1])
                        c1.markdown(f"**Net Rating: :green[{row['NET_RATING']:+.1f}]**")
                        c2.markdown(f"**Total Mins: {row['minutes']:.1f}**")
                        
                        # Progress bar with explicit label
                        share_pct = (row['minutes'] / max_mins) * 100
                        st.caption(f"Rotation Share (vs. Most Used Lineup): {share_pct:.0f}%")
                        st.progress(min(row['minutes'] / max_mins, 1.0))
                        
                        # Lineup Names
                        names_list = row['LINEUP_NAME'].replace('-', ' • ')
                        st.markdown(f"<div style='font-size: 0.85rem; color: #888;'>{names_list}</div>", unsafe_allow_html=True)

            with col_bot:
                st.subheader("⚠️ Bottom 5 Lineups")
                for _, row in bot_5.iterrows():
                    with st.container(border=True):
                        # Header: Rating and Minutes
                        c1, c2 = st.columns([1, 1])
                        c1.markdown(f"**Net Rating: :red[{row['NET_RATING']:+.1f}]**")
                        c2.markdown(f"**Total Mins: {row['minutes']:.1f}**")
                        
                        # Progress bar with explicit label
                        share_pct = (row['minutes'] / max_mins) * 100
                        st.caption(f"Rotation Share (vs. Most Used Lineup): {share_pct:.0f}%")
                        st.progress(min(row['minutes'] / max_mins, 1.0))
                        
                        # Lineup Names
                        names_list = row['LINEUP_NAME'].replace('-', ' • ')
                        st.markdown(f"<div style='font-size: 0.85rem; color: #888;'>{names_list}</div>", unsafe_allow_html=True)

   # --- 5. SCORING MOMENTUM & RESILIENCE ---
    st.divider()
    st.markdown(f"### 🌊 Scoring Momentum & Resilience: {team_blue}{filter_status}", unsafe_allow_html=True)

    # 1. UTILS: Percentile logic for runs
    def get_run_percentile(value, series, lower_is_better=False):
        if series.empty: return 0
        below = (series > value).sum() if lower_is_better else (series < value).sum()
        tied = (series == value).sum()
        return ((below + (0.5 * tied)) / len(series)) * 100

    # --- NEW: LEAGUE BENCHMARKING FOR RUNS ---
    all_teams_list = df_teams[['TEAM_NAME', 'CONFERENCE', 'CONFERENCE_TYPE']].copy()
    
    # League Runs For
    run_counts_for = df_scoring_runs[df_scoring_runs['TOTAL_RUN_POINTS'] >= 10].groupby('TEAM_ON_RUN').size().reset_index(name='run_count')
    league_runs_for = all_teams_list.merge(run_counts_for, left_on='TEAM_NAME', right_on='TEAM_ON_RUN', how='left').fillna(0)

    # League Runs Against
    runs_against_base = df_scoring_runs[df_scoring_runs['TOTAL_RUN_POINTS'] >= 10].merge(df_games[['GAME_ID', 'TEAM_NAME']], on='GAME_ID')
    # Count how many times each team was the "Victim" (i.e. opponent of the team on the run)
    runs_against_counts = runs_against_base[runs_against_base['TEAM_ON_RUN'] != runs_against_base['TEAM_NAME']].groupby('TEAM_NAME').size().reset_index(name='gave_up_count')
    league_runs_against = all_teams_list.merge(runs_against_counts, on='TEAM_NAME', how='left').fillna(0)
    # ------------------------------------------

    # 2. FILTERED TEAM DATA (Consistent with Power Toggle)
    runs_for = df_scoring_runs[(df_scoring_runs['TEAM_ON_RUN'] == selected_team) & (df_scoring_runs['GAME_ID'].isin(valid_game_ids)) & (df_scoring_runs['TOTAL_RUN_POINTS'] >= 10)].copy()
    runs_against = df_scoring_runs[(df_scoring_runs['GAME_ID'].isin(valid_game_ids)) & (df_scoring_runs['TEAM_ON_RUN'] != selected_team) & (df_scoring_runs['TOTAL_RUN_POINTS'] >= 10)].copy()

    # 3. UI LAYOUT
    col_made, col_given = st.columns(2)

    with col_made:
        val_for = len(runs_for)
        p_g_for = get_run_percentile(val_for, league_runs_for['run_count'])
        p_c_for = get_run_percentile(val_for, league_runs_for[league_runs_for['CONFERENCE'] == conf]['run_count'])
        p_t_for = get_run_percentile(val_for, league_runs_for[league_runs_for['CONFERENCE_TYPE'] == tier]['run_count'])
        draw_card("10+ Pt Runs Made", f"{val_for}", p_g_for, p_c_for, p_t_for, conf)
        
        with st.container(border=True):
            counts_f = runs_for.groupby('GAME_ID').size().reset_index(name='c')
            win_f = team_games[['GAME_ID', 'TEAM_VICTORY_INDICATOR']].merge(counts_f, on='GAME_ID', how='left').fillna(0)
            win_f['Bin'] = win_f['c'].apply(lambda x: "0 Runs" if x == 0 else "1 Run" if x == 1 else "2+ Runs")
            
            f_stats = []
            ordered_f = ["0 Runs", "1 Run", "2+ Runs"]
            for b in ordered_f:
                sub = win_f[win_f['Bin'] == b]
                w, t = sub['TEAM_VICTORY_INDICATOR'].sum(), len(sub)
                pct = (w/t*100) if t > 0 else 0
                clr = "#28a745" if pct >= 75 else "#ffc107" if pct >= 51 else "#dc3545"
                f_stats.append({"Bin": b, "WinPct": pct, "Rec": f"{int(w)}-{int(t-w)}", "Color": clr})
            
            fig_f = px.bar(pd.DataFrame(f_stats), x="Bin", y="WinPct", text="Rec", color="Color", color_discrete_map="identity", template="plotly_dark", height=180)
            fig_f.update_layout(margin=dict(l=5,r=5,t=5,b=5), yaxis_range=[0,105], showlegend=False, xaxis={'categoryorder':'array', 'categoryarray': ordered_f, 'title': None})
            st.plotly_chart(fig_f, use_container_width=True, config={'displayModeBar': False})

    with col_given:
        val_against = len(runs_against)
        p_g_against = get_run_percentile(val_against, league_runs_against['gave_up_count'], lower_is_better=True)
        p_c_against = get_run_percentile(val_against, league_runs_against[league_runs_against['CONFERENCE'] == conf]['gave_up_count'], lower_is_better=True)
        p_t_against = get_run_percentile(val_against, league_runs_against[league_runs_against['CONFERENCE_TYPE'] == tier]['gave_up_count'], lower_is_better=True)
        draw_card("10+ Pt Runs Given Up", f"{val_against}", p_g_against, p_c_against, p_t_against, conf)
        
        with st.container(border=True):
            counts_a = runs_against.groupby('GAME_ID').size().reset_index(name='c')
            win_a = team_games[['GAME_ID', 'TEAM_VICTORY_INDICATOR']].merge(counts_a, on='GAME_ID', how='left').fillna(0)
            win_a['Bin'] = win_a['c'].apply(lambda x: "0 Given" if x == 0 else "1 Given" if x == 1 else "2+ Given")
            
            a_stats = []
            ordered_a = ["0 Given", "1 Given", "2+ Given"]
            for b in ordered_a:
                sub = win_a[win_a['Bin'] == b]
                w, t = sub['TEAM_VICTORY_INDICATOR'].sum(), len(sub)
                pct = (w/t*100) if t > 0 else 0
                clr = "#28a745" if pct >= 75 else "#ffc107" if pct >= 51 else "#dc3545"
                a_stats.append({"Bin": b, "WinPct": pct, "Rec": f"{int(w)}-{int(t-w)}", "Color": clr})
            
            fig_a = px.bar(pd.DataFrame(a_stats), x="Bin", y="WinPct", text="Rec", color="Color", color_discrete_map="identity", template="plotly_dark", height=180)
            fig_a.update_layout(margin=dict(l=5,r=5,t=5,b=5), yaxis_range=[0,105], showlegend=False, xaxis={'categoryorder':'array', 'categoryarray': ordered_a, 'title': None})
            st.plotly_chart(fig_a, use_container_width=True, config={'displayModeBar': False})

    # --- 6. UNIFIED MOMENTUM LOG ---
    st.write("")
    st.subheader("📅 Unified Momentum Log (10+ Point Runs)")
    runs_for_copy = runs_for.copy()
    runs_against_copy = runs_against.copy()
    runs_for_copy['Type'], runs_against_copy['Type'] = '🏃 GONE ON', '🛡️ SURRENDERED'
    
    all_big_runs = pd.concat([runs_for_copy, runs_against_copy])
    combined_log = all_big_runs.merge(team_games[['GAME_ID', 'GAME_DATE', 'OPPONENT_TEAM_NAME', 'TEAM_VICTORY_INDICATOR']], on='GAME_ID').sort_values(['GAME_DATE', 'RUN_START_CLOCK'], ascending=[False, False])
    combined_log['Res'], combined_log['GAME_DATE'] = combined_log['TEAM_VICTORY_INDICATOR'].apply(lambda x: "W" if x else "L"), pd.to_datetime(combined_log['GAME_DATE']).dt.date
    display_log = combined_log[['GAME_DATE', 'Type', 'OPPONENT_TEAM_NAME', 'Res', 'TOTAL_RUN_POINTS', 'NUM_SCORING_PLAYS_IN_RUN', 'RUN_START_CLOCK', 'RUN_END_CLOCK']]
    display_log.columns = ['Date', 'Type', 'Opponent', 'Result', 'Pts', 'Plays', 'Start', 'End']

    st.dataframe(display_log.style.apply(lambda row: ['background-color: rgba(59, 18, 245, 0.1)' if 'GONE' in str(row.Type) else 'background-color: rgba(255, 75, 75, 0.1)'] * len(row), axis=1), use_container_width=True, hide_index=True)

# --- 7. MOMENTUM QUADRANT SCATTERPLOT ---
    st.divider()
    st.markdown(f"### 🎯 Momentum Quadrant: {team_blue} vs League", unsafe_allow_html=True)
    
    # 1. Prepare Plotly Data
    # league_runs_for and league_runs_against were already calculated in the Resilience section
    scatter_df = league_runs_for.merge(league_runs_against[['TEAM_NAME', 'gave_up_count']], on='TEAM_NAME')
    
    # Calculate Win % for bubble size
    league_wins = df_games.groupby('TEAM_NAME')['TEAM_VICTORY_INDICATOR'].agg(['sum', 'count']).reset_index()
    league_wins['WinPct'] = (league_wins['sum'] / league_wins['count']) * 100
    scatter_df = scatter_df.merge(league_wins[['TEAM_NAME', 'WinPct']], on='TEAM_NAME')
    
    # Filter by conference if desired
    show_conf_only = st.checkbox(f"Focus on {conf} Conference Only", value=False)
    if show_conf_only:
        plot_df = scatter_df[scatter_df['CONFERENCE'] == conf].copy()
    else:
        plot_df = scatter_df.copy()

    # 2. Build Plot
    fig_scatter = px.scatter(
        plot_df,
        x="run_count",
        y="gave_up_count",
        size="WinPct",
        hover_name="TEAM_NAME",
        labels={
            "run_count": "10+ Pt Runs Made",
            "gave_up_count": "10+ Pt Runs Given Up"
        },
        template="plotly_dark",
        color_discrete_sequence=["#666666"], # Default color for league
        opacity=0.5
    )

    # 3. Highlight the Selected Team
    selected_data = scatter_df[scatter_df['TEAM_NAME'] == selected_team]
    if not selected_data.empty:
        fig_scatter.add_trace(
            go.Scatter(
                x=selected_data["run_count"],
                y=selected_data["gave_up_count"],
                mode="markers+text",
                marker=dict(color="#3B12F5", size=selected_data["WinPct"] * 0.8, line=dict(width=2, color='white')),
                text=[selected_team],
                textposition="top center",
                name=selected_team,
                hoverinfo="skip"
            )
        )

    # 4. Styling (Quadrants)
    # Average lines to create quadrants
    fig_scatter.add_vline(x=scatter_df['run_count'].mean(), line_dash="dot", line_color="white", opacity=0.3)
    fig_scatter.add_hline(y=scatter_df['gave_up_count'].mean(), line_dash="dot", line_color="white", opacity=0.3)
    
    fig_scatter.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        yaxis=dict(autorange="reversed") # Better teams give up FEWER runs, so we reverse Y
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.caption("💡 **Note:** The Y-axis is reversed. Top-Right represents the 'Elite' quadrant (Many runs made, few given up). Bubble size reflects overall Win %.")

# * ATO EFFICIENCY *
elif selection == "⏱️ After Timeout Efficiency":
    # Use the 'selected_team' defined globally in the sidebar
    team_blue = f"<span style='color: #3B12F5; text-decoration: underline; font-weight: bold;'>{selected_team}</span>"
    st.markdown(f"## ⏱️ After Timeout (ATO) Analysis: {team_blue}", unsafe_allow_html=True)

    # 1. DATA PREP
    df_ato_ui = df_ato.copy().rename(columns={
        'TEAM_NAME': 'Team',
        'CONFERENCE': 'Conference',
        'CONFERENCE_TYPE': 'Tier',
        'ATTEMPT_QUALITY': 'Shot Type',
        'POINTS_PER_PLAY': 'PPP',
        'PLAYS_RUN': 'Plays Run',
        'TOTAL_POINTS_SCORED': 'Total Points'
    })
    
    for col in ['PPP', 'Plays Run', 'Total Points']:
        df_ato_ui[col] = pd.to_numeric(df_ato_ui[col], errors='coerce').fillna(0).round(2)

    # Get metadata for the globally selected team
    try:
        team_info = df_ato_ui[df_ato_ui['Team'] == selected_team].iloc[0]
        target_conf = team_info['Conference']
        target_tier = team_info['Tier']
    except IndexError:
        st.error(f"Data for {selected_team} not found in the ATO dataset.")
        st.stop()
        
    conf_df = df_ato_ui[df_ato_ui['Conference'] == target_conf].copy()
    tier_df = df_ato_ui[df_ato_ui['Tier'] == target_tier].copy()

    # --- 2. RANKED PPP EFFICIENCY (BAR CHART) ---
    st.subheader(f"Ranked Total ATO Efficiency: {target_conf}")
    total_rank_df = conf_df.groupby('Team').agg({'Total Points': 'sum', 'Plays Run': 'sum'}).reset_index()
    total_rank_df['PPP'] = (total_rank_df['Total Points'] / total_rank_df['Plays Run']).round(2)
    total_rank_df = total_rank_df.sort_values('PPP', ascending=False)
    
    # Highlight the selected team in Blue
    total_rank_df['Color'] = total_rank_df['Team'].apply(lambda x: "#3B12F5" if x == selected_team else '#31333F')

    fig_rank = px.bar(
        total_rank_df, x='PPP', y='Team', orientation='h',
        text_auto='.2f', color='Color', color_discrete_map="identity",
        template="plotly_dark", height=max(400, len(total_rank_df)*25)
    )
    fig_rank.update_traces(hovertemplate="Team: %{y}<br>PPP: %{x}<extra></extra>")
    fig_rank.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    # --- 3. EFFICIENCY VS. VOLUME (SCATTERPLOT FIX) ---
    st.divider()
    st.subheader("Efficiency vs. Volume Analysis")
    
    view_option = st.selectbox("Select Shot Category", ["Total PPP", "Three Point", "Mid Range", "Attack Basket"])

    if view_option == "Total PPP":
        plot_df = total_rank_df.rename(columns={'PPP': 'y', 'Plays Run': 'x'})
        # Benchmarks
        g_avg, t_avg, c_avg = df_ato_ui['PPP'].mean(), tier_df['PPP'].mean(), conf_df['PPP'].mean()
    else:
        # Filter for specific shot type
        plot_df = conf_df[conf_df['Shot Type'] == view_option].copy().rename(columns={'PPP': 'y', 'Plays Run': 'x'})
        # Benchmarks for specific shot type
        g_avg = df_ato_ui[df_ato_ui['Shot Type'] == view_option]['PPP'].mean()
        t_avg = tier_df[tier_df['Shot Type'] == view_option]['PPP'].mean()
        c_avg = conf_df[conf_df['Shot Type'] == view_option]['PPP'].mean()

    # Create Scatter
    fig_scatter = px.scatter(
        plot_df, x='x', y='y', 
        color='Team', # This automatically maps team names to the trace
        size='x',
        text='Team',
        labels={'x': 'Plays Run', 'y': 'PPP'},
        title=f"{view_option} Benchmarking: {target_conf}", 
        template="plotly_dark"
    )

    # Use %{fullData.name} or simply remove customdata to let Plotly use the 'color' label
    fig_scatter.update_traces(
        textposition='top center',
        hovertemplate="<b>%{text}</b><br>Plays: %{x}<br>PPP: %{y:.2f}<extra></extra>"
    )
    
    # Add Horizontal benchmark lines
    line_configs = [
        ("NCAA Avg", g_avg, "white", "dot"), 
        (f"{target_tier} Avg", t_avg, "#00CC96", "dash"), 
        (f"{target_conf} Avg", c_avg, "#3B12F5", "solid")
    ]
    
    for label, val, color, style in line_configs:
        fig_scatter.add_hline(y=val, line_dash=style, line_color=color, 
                              annotation_text=f" {label}: {val:.2f}", annotation_position="top right")

    fig_scatter.update_layout(showlegend=False) # Hide legend since names are on plot
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 4. STRATEGY DNA (STACKED BAR) ---
    st.divider()
    st.subheader("ATO Shot Selection Profile")
    st.caption("How teams choose to attack after a timeout (Percentage of Total ATO Plays)")
    
    # Calculate DNA Percentages
    dna_df = conf_df.copy()
    dna_df['Total Plays'] = dna_df.groupby('Team')['Plays Run'].transform('sum')
    dna_df['Pct'] = ((dna_df['Plays Run'] / dna_df['Total Plays']) * 100).round(1)

    # Order by the ranking calculated in Step 2
    dna_sort_order = total_rank_df['Team'].tolist()#[::-1]

    fig_stack = px.bar(
        dna_df, y="Team", x="Pct", color="Shot Type",
        text="Pct", orientation='h',
        title=f"Strategy DNA: {target_conf}",
        labels={"Pct": "Selection %"},
        template="plotly_dark",
        category_orders={"Team": dna_sort_order},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig_stack.update_traces(texttemplate='%{text}%', 
                            hovertemplate="Team: %{y}<br>Percentage: %{x}%<extra></extra>",
                            textposition='inside')
    fig_stack.update_layout(xaxis_ticksuffix="%", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_stack, use_container_width=True)

# * MOMENTUM & ADJUSTMENTS *
elif selection == "🔥 Momentum & Adjustments":
    team_info = df_teams[df_teams['TEAM_NAME'] == selected_team].iloc[0]
    conf, tier = team_info['CONFERENCE'], team_info['CONFERENCE_TYPE']
    team_blue = f"<span style='color: #3B12F5; text-decoration: underline; font-weight: bold;'>{selected_team}</span>"
    st.markdown(f"## 🔥 {team_blue}: Momentum & Adjustments", unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* This targets the vertical block immediately following our label */
        [data-testid="stVerticalBlock"] > div:has(div > p:contains("Win Prob by")) {
            border: 1px solid #444 !important;
            border-radius: 10px !important;
            background-color: #1e1e1e !important;
            padding: 10px !important;
            margin-bottom: 15px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def clock_to_minute(clock_str):
        try:
            parts = clock_str.split('-')
            period, mins = int(parts[0]), int(parts[1].split(':')[0])
            return max(1, ((period - 1) * 20) + (20 - mins))
        except: return None

    # Offense
    c_df = df_crushers[df_crushers['TEAM_NAME'] == selected_team].copy()
    c_df['GAME_MINUTE'] = c_df['GAME_CLOCK_TIME'].apply(clock_to_minute)
    league_c = df_crushers.groupby('TEAM_NAME')['OFFENSIVE_CRUSH_INDICATOR'].sum().reset_index()
    
    # Defense
    k_df = df_kills[df_kills['TEAM_NAME'] == selected_team].copy()
    k_df['GAME_MINUTE'] = k_df['GAME_CLOCK_TIME'].apply(clock_to_minute)
    league_k = df_kills.groupby('TEAM_NAME')['DEFENSIVE_KILL_INDICATOR'].sum().reset_index()

    # Shared Game Win Logic
    win_logic = df_games[df_games['TEAM_NAME'] == selected_team][['GAME_ID', 'TEAM_VICTORY_INDICATOR']]

    col_off, col_def = st.columns(2)

    # --- OFFENSIVE SIDE ---
    with col_off:
        st.subheader("🏀 Offensive Crushers")
        
        val_c = int(c_df['OFFENSIVE_CRUSH_INDICATOR'].sum())
        p_g_c = (league_c['OFFENSIVE_CRUSH_INDICATOR'] < val_c).mean() * 100
        p_c_c = (league_c[league_c['TEAM_NAME'].isin(df_teams[df_teams['CONFERENCE'] == conf]['TEAM_NAME'])]['OFFENSIVE_CRUSH_INDICATOR'] < val_c).mean() * 100
        p_t_c = (league_c[league_c['TEAM_NAME'].isin(df_teams[df_teams['CONFERENCE_TYPE'] == tier]['TEAM_NAME'])]['OFFENSIVE_CRUSH_INDICATOR'] < val_c).mean() * 100
        draw_card("Total Offensive Crushers", f"{val_c}", p_g_c, p_c_c, p_t_c, conf)

        game_c = c_df.groupby('GAME_ID')['OFFENSIVE_CRUSH_INDICATOR'].sum().reset_index()
        m_c = game_c.merge(win_logic, on='GAME_ID')
        m_c['Bin'] = m_c['OFFENSIVE_CRUSH_INDICATOR'].apply(lambda x: "0" if x==0 else "1-3" if x<=3 else "4-5" if x<=5 else "6+")
        
        bin_res_c = []
        for b in ["0", "1-3", "4-5", "6+"]:
            sub = m_c[m_c['Bin'] == b]
            w = sub['TEAM_VICTORY_INDICATOR'].sum()
            t = len(sub)
            pct = (w/t*100) if t > 0 else 0
            color = "#28a745" if pct >= 75 else "#ffc107" if pct >= 50 else "#dc3545"
            bin_res_c.append({"Bin": b, "WinPct": pct, "Record": f"{int(w)}-{int(t-w)}", "Color": color})
        
        fig_c = px.bar(pd.DataFrame(bin_res_c), x="Bin", y="WinPct", text="Record", color="Color", color_discrete_map="identity", template="plotly_dark", height=150)
        fig_c.update_layout(margin=dict(l=5,r=5,t=20,b=5), xaxis_title=None, yaxis_visible=False, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_c.update_traces(hovertemplate="<b>Crushers: %{x}</b><br>Record: %{text} (%{y:.0f}%)<extra></extra>")
        
        with st.container():
            st.markdown("<div style='color:#bbb; font-size:0.8rem; text-align:center;'>Win Prob by Crusher Count</div>", unsafe_allow_html=True)
            fig_c.update_layout(
                margin=dict(l=5, r=5, t=25, b=5), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

        c_trend = c_df[c_df['OFFENSIVE_CRUSH_INDICATOR'] == 1].groupby('GAME_MINUTE').size().reset_index(name='count')
        fig_wf_c = px.bar(c_trend, x='GAME_MINUTE', y='count', template="plotly_dark", color_discrete_sequence=['#3B12F5'], height=250)
        fig_wf_c.update_layout(xaxis=dict(range=[0,42]), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_wf_c, use_container_width=True)

    # --- DEFENSIVE SIDE ---
    with col_def:
        st.subheader("🛡️ Defensive Kills")
        
        val_k = int(k_df['DEFENSIVE_KILL_INDICATOR'].sum())
        p_g_k = (league_k['DEFENSIVE_KILL_INDICATOR'] < val_k).mean() * 100
        p_c_k = (league_k[league_k['TEAM_NAME'].isin(df_teams[df_teams['CONFERENCE'] == conf]['TEAM_NAME'])]['DEFENSIVE_KILL_INDICATOR'] < val_k).mean() * 100
        p_t_k = (league_k[league_k['TEAM_NAME'].isin(df_teams[df_teams['CONFERENCE_TYPE'] == tier]['TEAM_NAME'])]['DEFENSIVE_KILL_INDICATOR'] < val_k).mean() * 100
        draw_card("Total Defensive Kills", f"{val_k}", p_g_k, p_c_k, p_t_k, conf)

        game_k = k_df.groupby('GAME_ID')['DEFENSIVE_KILL_INDICATOR'].sum().reset_index()
        m_k = game_k.merge(win_logic, on='GAME_ID')
        m_k['Bin'] = m_k['DEFENSIVE_KILL_INDICATOR'].apply(lambda x: "0" if x==0 else "1-3" if x<=3 else "4-6" if x<=6 else "7+")
        
        bin_res_k = []
        for b in ["0", "1-3", "4-6", "7+"]:
            sub = m_k[m_k['Bin'] == b]
            w = sub['TEAM_VICTORY_INDICATOR'].sum()
            t = len(sub)
            pct = (w/t*100) if t > 0 else 0
            color = "#28a745" if pct >= 75 else "#ffc107" if pct >= 50 else "#dc3545"
            bin_res_k.append({"Bin": b, "WinPct": pct, "Record": f"{int(w)}-{int(t-w)}", "Color": color})
        
        fig_k = px.bar(pd.DataFrame(bin_res_k), x="Bin", y="WinPct", text="Record", color="Color", color_discrete_map="identity", template="plotly_dark", height=150)
        fig_k.update_layout(margin=dict(l=5,r=5,t=20,b=5), xaxis_title=None, yaxis_visible=False, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_k.update_traces(hovertemplate="<b>Kills: %{x}</b><br>Record: %{text} (%{y:.0f}%)<extra></extra>")

        with st.container():
            st.markdown("<div style='color:#bbb; font-size:0.8rem; text-align:center;'>Win Prob by Kill Count</div>", unsafe_allow_html=True)
            fig_k.update_layout(
                margin=dict(l=5, r=5, t=25, b=5), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_k, use_container_width=True, config={'displayModeBar': False})

        k_trend = k_df[k_df['DEFENSIVE_KILL_INDICATOR'] == 1].groupby('GAME_MINUTE').size().reset_index(name='count')
        fig_wf_k = px.bar(k_trend, x='GAME_MINUTE', y='count', template="plotly_dark", color_discrete_sequence=['#FF4B4B'], height=250)
        fig_wf_k.update_layout(xaxis=dict(range=[0,42]), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_wf_k, use_container_width=True)

    # --- 3. THE HALFTIME PIVOT ---
    st.divider()
    st.markdown("### 🔄 Halftime Pivot: Adjustment Scatter")
        
    scope = st.radio("View Scope:", ["All NCAA", f"{conf} Conference Only"], horizontal=True, label_visibility="collapsed", key="pivot_scope_unique")

    df_pivot_all = df_crushers.copy()
    df_pivot_all['GAME_MINUTE'] = df_pivot_all['GAME_CLOCK_TIME'].apply(clock_to_minute)
    df_pivot_all = df_pivot_all[df_pivot_all['GAME_MINUTE'].between(17, 24)]
    df_pivot_all['window'] = np.where(df_pivot_all['GAME_MINUTE'] <= 20, 'End_1H', 'Start_2H')
    
    league_pivot = df_pivot_all.groupby(['TEAM_NAME', 'window'])['POSSESSION_RESULT_SCORE'].mean().unstack().reset_index()
    league_pivot.columns = ['Team', 'End_1H', 'Start_2H']
    league_pivot = league_pivot.merge(df_teams[['TEAM_NAME', 'CONFERENCE', 'CONFERENCE_TYPE']], left_on='Team', right_on='TEAM_NAME').dropna()

    if "Conference Only" in scope:
        plot_df = league_pivot[league_pivot['CONFERENCE'] == conf].copy()
        color_col = None 
    else:
        plot_df = league_pivot.copy()
        color_col = 'CONFERENCE_TYPE'

    fig_scatter = px.scatter(
        plot_df, x='End_1H', y='Start_2H',
        color=color_col,
        hover_name='Team',
        hover_data={'CONFERENCE': True, 'End_1H': ':.1%', 'Start_2H': ':.1%'},
        template="plotly_dark",
        labels={'End_1H': 'Final 4m (1H) Scoring %', 'Start_2H': 'First 4m (2H) Scoring %'},
        opacity=0.75,
        color_discrete_map={'Power': "#F3A30F", 'Mid-Major': "#72D3F3"} # Match your data labels here
    )
    
    team_dot = league_pivot[league_pivot['Team'] == selected_team]
    if not team_dot.empty:
        fig_scatter.add_trace(go.Scatter(
            x=team_dot['End_1H'], 
            y=team_dot['Start_2H'], 
            mode='markers+text',
            marker=dict(color="#3B12F5", size=18, symbol='star', line=dict(color='white', width=2)),
            text=[selected_team],
            textposition="top center",
            name=selected_team,
            showlegend=False
        ))

    max_val = max(plot_df['End_1H'].max(), plot_df['Start_2H'].max()) if not plot_df.empty else 1
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="white", dash="dash", width=1))
    fig_scatter.add_annotation(x=max_val, y=max_val, text="Improved After Half ↗", showarrow=False, yshift=10, font=dict(color="#28a745"))
    fig_scatter.add_annotation(x=max_val, y=0, text="Faded After Half ↘", showarrow=False, yshift=-10, font=dict(color="#dc3545"))
    fig_scatter.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )    
    st.plotly_chart(fig_scatter, use_container_width=True)
