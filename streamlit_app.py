import streamlit as st
import pandas as pd
import plotly.express as px
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# ---------------------------------------------------------
# 1. PAGE CONFIG & AUTH (Same as before)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. DATA CACHING LAYER
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_all_data():
    conn = st.connection("snowflake", private_key=get_private_key())
    
    # Existing dimension table
    df_teams = conn.query("SELECT * FROM cbb_data.views.dim_team")
    
    # New ATO Fact table
    df_ato = conn.query("SELECT * FROM cbb_data.views.fact_ato_results")
    
    return df_teams, df_ato

# Unpack both dataframes
df_teams, df_ato = load_all_data()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Select a Report", ["🏠 Home", "📊 Team Analysis", "⏱️ After Timeout Efficiency"],
        key="main_nav"
    )
    
    st.divider()
    st.info("Data cached from Snowflake.")

# ---------------------------------------------------------
# 4. REPORT DISPLAY LOGIC
# ---------------------------------------------------------

# --- HOME PAGE ---
if selection == "🏠 Home":
    st.title("🏆 Snowflake Sports Dashboard")
    st.header("Welcome")
    st.markdown("Select a report from the sidebar to begin.")

# --- TEAM ANALYSIS ---
elif selection == "📊 Team Analysis":
    st.title("📊 Team Analysis")
    
    # Nested filters stay inside the "if" block so they only exist here
    team_list = sorted(df_teams['TEAM_NAME'].unique())
    selected_team = st.selectbox("Select Team", team_list, key="team_sel")
    
    filtered_df = df_teams[df_teams['TEAM_NAME'] == selected_team]
    st.dataframe(filtered_df, use_container_width=True)

elif selection == "⏱️ After Timeout Efficiency":
    st.title("⏱️ After Timeout (ATO) Efficiency Comparisons")

    # --- 1. DATA PREP & GLOBAL UI RENAMING ---
    df_ato_ui = df_ato.copy().rename(columns={
        'TEAM_NAME': 'Team',
        'CONFERENCE': 'Conference',
        'CONFERENCE_TYPE': 'Tier',
        'ATTEMPT_QUALITY': 'Shot Type',
        'POINTS_PER_PLAY': 'PPP',
        'PLAYS_RUN': 'Plays Run',
        'TOTAL_POINTS_SCORED': 'Total Points'
    })
    
    # Cast and round globally
    for col in ['PPP', 'Plays Run', 'Total Points']:
        df_ato_ui[col] = df_ato_ui[col].astype(float).round(2)

    # --- 2. FILTERS ---
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

    # --- 3. RANKED TOTAL EFFICIENCY ---
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

    # --- 4. SCATTERPLOT: LABELS & BENCHMARKS ---
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

    # Added 'text' argument for team name labels
    fig_scatter = px.scatter(
        plot_df, x='x', y='y', color='Team', size='x',
        text='Team', # <--- Labels added here
        labels={'x': 'Plays Run', 'y': 'PPP'},
        title=f"{view_option} Benchmarking", template="plotly_dark"
    )

    # Styling the text labels so they don't overlap the bubbles
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

    # --- 5. STRATEGY DNA (SORTED BY TOTAL PPP) ---
    st.divider()
    st.subheader("ATO Shot Selection Profile")
    st.caption("Sorted by Total Team Efficiency (Highest PPP at the top)")
    
    # Calculate percentages
    conf_df['Total Plays'] = conf_df.groupby('Team')['Plays Run'].transform('sum')
    conf_df['Pct'] = ((conf_df['Plays Run'] / conf_df['Total Plays']) * 100).round(1)

    # Logic to sort the DNA chart by the Total Efficiency from total_rank_df
    # We create a list of team names in the order of their total PPP
    dna_sort_order = total_rank_df['Team'].tolist()[::-1]

    fig_stack = px.bar(
        conf_df, y="Team", x="Pct", color="Shot Type",
        text="Pct", orientation='h',
        title=f"Strategy DNA: {target_conf}",
        labels={"Pct": "Selection %"},
        template="plotly_dark",
        category_orders={"Team": dna_sort_order} # <--- Sorting logic
    )
    
    fig_stack.update_traces(texttemplate='%{text}%', 
                            hovertemplate="Team: %{y}<br>Percentage: %{x}<extra></extra>",
                            textposition='inside')
    fig_stack.update_layout(xaxis_ticksuffix="%", yaxis={'categoryorder':'array', 'categoryarray': dna_sort_order})
    st.plotly_chart(fig_stack, use_container_width=True)
