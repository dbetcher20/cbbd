import streamlit as st
import pandas as pd
import plotly.express as px
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import numpy as np
import plotly.graph_objects as go 


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
    df_crushers = conn.query("SELECT * FROM cbb_data.views.fact_consecutive_possessions_offensive_crushers")
    df_kills = conn.query("SELECT * FROM cbb_data.views.fact_consecutive_possessions_defensive_kills")
    return df_teams, df_ato, df_games, df_crushers, df_kills
df_teams, df_ato, df_games, df_crushers, df_kills = load_all_data()

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
    
    # Base dataset for the selected team
    raw_team_games = df_games[df_games['TEAM_NAME'] == selected_team].copy()
    
    st.markdown(f"## 📊 {selected_team} Performance Breakdown")
    
    # Master Toggle for Report Context
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

    # Sort for calculations (Ascending for running record)
    team_games = team_games.sort_values('GAME_DATE', ascending=True)
    
    # Calculate Running Record based on CURRENT filter
    team_games['Wins_Cum'] = team_games['TEAM_VICTORY_INDICATOR'].cumsum()
    team_games['Losses_Cum'] = (~team_games['TEAM_VICTORY_INDICATOR']).cumsum()
    team_games['Record_Str'] = team_games.apply(
        lambda x: f"{'W' if x['TEAM_VICTORY_INDICATOR'] else 'L'} ({int(x['Wins_Cum'])}-{int(x['Losses_Cum'])})", axis=1
    )

    # --- 2. PERCENTILE ENGINE & UTILS ---
    team_info = df_teams[df_teams['TEAM_NAME'] == selected_team].iloc[0]
    conf, tier = team_info['CONFERENCE'], team_info['CONFERENCE_TYPE']
    team_blue = f"<span style='color: #3B12F5; text-decoration: underline; font-weight: bold;'>{selected_team}</span>"

    stat_cols = ['TEAM_POINTS', 'TEAM_2PT_FG_PERCENT', 'TEAM_3PT_FG_PERCENT', 'TEAM_FT_PERCENT', 'TEAM_EFFECTIVE_FG_PERCENT',
                 'OPPONENT_POINTS', 'OPPONENT_2PT_FG_PERCENT', 'OPPONENT_3PT_FG_PERCENT', 'OPPONENT_FT_PERCENT', 'OPPONENT_EFFECTIVE_FG_PERCENT']
    
    all_teams_avg = df_games.groupby(['TEAM_NAME', 'CONFERENCE', 'CONFERENCE_TYPE'])[stat_cols].mean().reset_index()

    def get_pct(col, val, group_df=all_teams_avg):
        if group_df.empty: return 0
        dist = group_df[col]
        return (dist > val).mean() * 100 if "OPPONENT" in col else (dist < val).mean() * 100

    # --- 3. OFFENSIVE & DEFENSIVE PROFILES ---
    st.markdown(f"### Performance Profile {filter_status}", unsafe_allow_html=True)
    
    # Offensive Section
    st.subheader("🏀 Offensive Profile")
    off_stats = [("Points", "TEAM_POINTS", ".1f"), ("2PT %", "TEAM_2PT_FG_PERCENT", ".1%"),
                 ("3PT %", "TEAM_3PT_FG_PERCENT", ".1%"), ("FT %", "TEAM_FT_PERCENT", ".1%"),
                 ("eFG %", "TEAM_EFFECTIVE_FG_PERCENT", ".1%")]
    c_off = st.columns(5)
    for i, (lab, col, fmt) in enumerate(off_stats):
        val = team_games[col].mean()
        p_g = get_pct(col, val)
        p_c = get_pct(col, val, all_teams_avg[all_teams_avg['CONFERENCE'] == conf])
        p_t = get_pct(col, val, all_teams_avg[all_teams_avg['CONFERENCE_TYPE'] == tier])
        with c_off[i]: draw_card(lab, f"{val:{fmt}}", p_g, p_c, p_t, conf)

    # Defensive Section
    st.write("")
    st.subheader("🛡️ Defensive Profile")
    def_stats = [("Pts Allowed", "OPPONENT_POINTS", ".1f"), ("Opp 2PT %", "OPPONENT_2PT_FG_PERCENT", ".1%"),
                 ("Opp 3PT %", "OPPONENT_3PT_FG_PERCENT", ".1%"), ("Opp FT %", "OPPONENT_FT_PERCENT", ".1%"),
                 ("Opp eFG %", "OPPONENT_EFFECTIVE_FG_PERCENT", ".1%")]
    c_def = st.columns(5)
    for i, (lab, col, fmt) in enumerate(def_stats):
        val = team_games[col].mean()
        p_g = get_pct(col, val)
        p_c = get_pct(col, val, all_teams_avg[all_teams_avg['CONFERENCE'] == conf])
        p_t = get_pct(col, val, all_teams_avg[all_teams_avg['CONFERENCE_TYPE'] == tier])
        with c_def[i]: draw_card(lab, f"{val:{fmt}}", p_g, p_c, p_t, conf)

    # --- 4. TRENDS ---
    st.divider()
    st.markdown(f"### 📈 Trends: {team_blue} vs Opponents{filter_status}", unsafe_allow_html=True)
    
    suffix = st.selectbox("Select Statistic", ['POINTS', 'TURNOVERS', 'STEALS', 'BLOCKS', 'REBOUNDS_OFF', 'REBOUNDS_DEF'])
    t_c, o_c = f"TEAM_{suffix}", f"OPPONENT_{suffix}"
    
    plot_df = team_games.copy()
    for col in [t_c, o_c]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0)

    metric_label = suffix.replace('_', ' ').title()

    import plotly.graph_objects as go
    import numpy as np
    fig = go.Figure()

    # Team Trace
    fig.add_trace(go.Scatter(
        x=plot_df['GAME_DATE'], y=plot_df[t_c],
        mode='lines+markers', name=selected_team,
        line=dict(color="#3B12F5", width=3),
        customdata=np.stack((plot_df['OPPONENT_TEAM_NAME'], plot_df[o_c]), axis=-1),
        hovertemplate=f"<b>%{{x}} - %{{customdata[0]}}</b><br>{metric_label}: %{{y}}<br>Opponent {metric_label}: %{{customdata[1]}}<extra></extra>"
    ))

    # Opponent Trace
    fig.add_trace(go.Scatter(
        x=plot_df['GAME_DATE'], y=plot_df[o_c],
        mode='lines+markers', name="Opponent",
        line=dict(color="#FF4B4B", width=3), hoverinfo='skip' 
    ))

    # Safe Trendlines
    if len(plot_df) > 1:
        for col, color, lab in [(t_c, "#3B12F5", selected_team), (o_c, "#FF4B4B", "Opponent")]:
            y_vals = plot_df[col].values
            x_vals = np.arange(len(y_vals))
            try:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(x=plot_df['GAME_DATE'], y=p(x_vals), mode='lines',
                                         line=dict(dash='dot', color=color, width=1.5), showlegend=False, hoverinfo='skip'))
            except: continue

    fig.update_layout(template="plotly_dark", hovermode="x", xaxis_title="Game Date", yaxis_title=metric_label,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. GAME LOG ---
    st.divider()
    st.subheader(f"📅 Game Log{filter_status}")
    log_final = team_games.sort_values('GAME_DATE', ascending=False)[['GAME_DATE', 'OPPONENT_TEAM_NAME', 'Record_Str', 'TEAM_POINTS', 'OPPONENT_POINTS']]
    log_final['GAME_DATE'] = log_final['GAME_DATE'].dt.date
    log_final.columns = ['Date', 'Opponent', 'Result', 'Score', 'Opp Score']
    
    def style_row(row):
        color = 'background-color: rgba(0,255,0,0.1)' if 'W' in str(row.Result) else 'background-color: rgba(255,0,0,0.1)'
        return [color]*len(row)
    
    st.dataframe(log_final.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)

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
