import streamlit as st
import pandas as pd
import plotly.express as px
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# PAGE CONFIG & AUTH
st.set_page_config(page_title="Snowflake Sports Analytics", layout="wide")

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
    return df_teams, df_ato, df_games
df_teams, df_ato, df_games = load_all_data()

team_list = sorted(df_teams['TEAM_NAME'].unique())
try:
    default_ix = team_list.index("Duke")
except ValueError:
    default_ix = 0

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("Reports")
    selection = st.radio("Select an Analytics Report", 
            ["🏠 Home", 
             "📊 Team Breakdown", 
             "⏱️ After Timeout Efficiency"
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
    st.title(f"📊 {selected_team} Performance Breakdown")
    
    team_games = df_games[df_games['TEAM_NAME'] == selected_team].copy()
    power_toggle = st.toggle("Filter by Power Conference Opponents")
    if power_toggle:
        analysis_df = team_games[team_games['OPPONENT_CONFERENCE_TYPE'] == 'Power']
    else:
        analysis_df = team_games

    # * KPI BANS *
    col1, col2, col3, col4 = st.columns(4)
    avg_pts = analysis_df['TEAM_POINTS'].mean()
    win_pts = analysis_df[analysis_df['TEAM_VICTORY_INDICATOR'] == True]['TEAM_POINTS'].mean()
    loss_pts = analysis_df[analysis_df['TEAM_VICTORY_INDICATOR'] == False]['TEAM_POINTS'].mean()

    col1.metric("Avg Points", f"{avg_pts:.1f}")
    st.caption(f"Wins: {win_pts:.1f} | Losses: {loss_pts:.1f}")
    
    # ... Add more metrics (Steals, Blocks, etc.) similarly ...

    # TREND LINE WITH DROPDOWN
    st.divider()
    trend_stat = st.selectbox("Select Trend Statistic", 
                               ['TEAM_POINTS', 'TEAM_TURNOVERS', 'TEAM_EFFECTIVE_FG_PERCENT'])
    
    fig_trend = px.line(team_games, x='GAME_DATE', y=[trend_stat, f'{trend_stat}'],
                        title=f"{trend_stat} Trend vs Opponents",
                        labels={'value': 'Stat Value', 'variable': 'Team'},
                        template="plotly_dark")
    st.plotly_chart(fig_trend, use_container_width=True)

# * ATO EFFICIENCY *
elif selection == "⏱️ After Timeout Efficiency":
    st.title("⏱️ After Timeout (ATO) Efficiency Comparisons")

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
        df_ato_ui[col] = df_ato_ui[col].astype(float).round(2)

    team_list = sorted(df_ato_ui['Team'].unique())
    try:
        default_ix = team_list.index("Duke") 
    except ValueError:
        default_ix = 0
    selected_team = st.selectbox("Select Primary Team", team_list, index=default_ix, key="ato_main_team")
    
    team_info = df_ato_ui[df_ato_ui['Team'] == selected_team].iloc[0]
    target_conf = team_info['Conference']
    target_tier = team_info['Tier']    
    conf_df = df_ato_ui[df_ato_ui['Conference'] == target_conf].copy()
    tier_df = df_ato_ui[df_ato_ui['Tier'] == target_tier].copy()

    # * RANKED PPP EFFICIENCY - bar *
    st.subheader(f"Ranked Total ATO Efficiency: {target_conf}")
    total_rank_df = conf_df.groupby('Team').agg({'Total Points': 'sum', 'Plays Run': 'sum'}).reset_index()
    total_rank_df['PPP'] = (total_rank_df['Total Points'] / total_rank_df['Plays Run']).round(2)
    total_rank_df = total_rank_df.sort_values('PPP', ascending=False)
    total_rank_df['Color'] = total_rank_df['Team'].apply(lambda x: "#3B12F5" if x == selected_team else '#31333F')

    fig_rank = px.bar(
        total_rank_df, x='PPP', y='Team', orientation='h',
        text_auto='.2f', color='Color', color_discrete_map="identity",
        template="plotly_dark"
    )
    fig_rank.update_traces(hovertemplate="Team: %{y}<br>PPP: %{x}<extra></extra>")
    fig_rank.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    # * PPP BY TEAM - scatterplot *
    st.divider()
    st.subheader("Efficiency vs. Volume Analysis")
    
    view_option = st.selectbox("Select Shot Category", ["Total PPP", "Three Point", "Mid Range", "Attack Basket"])

    if view_option == "Total PPP":
        plot_df = total_rank_df.rename(columns={'PPP': 'y', 'Plays Run': 'x'})
        g_avg, t_avg, c_avg = df_ato_ui['PPP'].mean(), tier_df['PPP'].mean(), conf_df['PPP'].mean()
    else:
        plot_df = conf_df[conf_df['Shot Type'] == view_option].copy().rename(columns={'PPP': 'y', 'Plays Run': 'x'})
        g_avg = df_ato_ui[df_ato_ui['Shot Type'] == view_option]['PPP'].mean()
        t_avg = tier_df[df_ato_ui['Shot Type'] == view_option]['PPP'].mean()
        c_avg = conf_df[df_ato_ui['Shot Type'] == view_option]['PPP'].mean()

    fig_scatter = px.scatter(
        plot_df, x='x', y='y', color='Team', size='x',
        text='Team',
        labels={'x': 'Plays Run', 'y': 'PPP'},
        title=f"{view_option} Benchmarking", template="plotly_dark"
    )

    fig_scatter.update_traces(
        textposition='top center',
        hovertemplate="Team: %{customdata[0]}<br>Plays: %{x}<br>PPP: %{y}<extra></extra>",
        customdata=plot_df[['Team']]
    )
    
    colors = {"Global": "white", "Tier": "#00CC96", "Conf": "#3B12F5"}
    avgs = [("Global", g_avg, "dot"), (f"{target_tier}", t_avg, "dash"), ("Conference", c_avg, "solid")]
    
    for label, val, style in avgs:
        fig_scatter.add_hline(y=val, line_dash=style, line_color=colors.get(label.split()[0], "white"), 
                              annotation_text=f" {label}: {val:.2f}", annotation_position="top right")

    st.plotly_chart(fig_scatter, use_container_width=True)

    # * PPP COMPOSITION - bar *
    st.divider()
    st.subheader("ATO Shot Selection Profile")
    st.caption("Sorted by Total Team Efficiency (Highest PPP at the top)")
    
    conf_df['Total Plays'] = conf_df.groupby('Team')['Plays Run'].transform('sum')
    conf_df['Pct'] = ((conf_df['Plays Run'] / conf_df['Total Plays']) * 100).round(1)

    dna_sort_order = total_rank_df['Team'].tolist()[::-1]

    fig_stack = px.bar(
        conf_df, y="Team", x="Pct", color="Shot Type",
        text="Pct", orientation='h',
        title=f"Strategy DNA: {target_conf}",
        labels={"Pct": "Selection %"},
        template="plotly_dark",
        category_orders={"Team": dna_sort_order}
    )
    
    fig_stack.update_traces(texttemplate='%{text}%', 
                            hovertemplate="Team: %{y}<br>Percentage: %{x}<extra></extra>",
                            textposition='inside')
    fig_stack.update_layout(xaxis_ticksuffix="%", yaxis={'categoryorder':'array', 'categoryarray': dna_sort_order})
    st.plotly_chart(fig_stack, use_container_width=True)
