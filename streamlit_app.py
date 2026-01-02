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

    # --- 1. DATA PREP & CLEANING ---
    # Create a UI-friendly copy for this tab
    df_ato_ui = df_ato.copy()
    df_ato_ui['POINTS_PER_PLAY'] = df_ato_ui['POINTS_PER_PLAY'].astype(float)
    df_ato_ui['PLAYS_RUN'] = df_ato_ui['PLAYS_RUN'].astype(float)
    df_ato_ui['TOTAL_POINTS_SCORED'] = df_ato_ui['TOTAL_POINTS_SCORED'].astype(float)

    # --- 2. FILTERS ---
    team_list = sorted(df_ato_ui['TEAM_NAME'].unique())
    selected_team = st.selectbox("Select Primary Team", team_list, key="ato_main_team")
    
    # Identify context for benchmarking
    team_info = df_ato_ui[df_ato_ui['TEAM_NAME'] == selected_team].iloc[0]
    target_conf = team_info['CONFERENCE']
    target_tier = team_info['CONFERENCE_TYPE']
    
    # Filter datasets for charts
    conf_df = df_ato_ui[df_ato_ui['CONFERENCE'] == target_conf].copy()
    tier_df = df_ato_ui[df_ato_ui['CONFERENCE_TYPE'] == target_tier].copy()

    # --- 3. DYNAMIC SCATTERPLOT WITH TRIPLE BENCHMARKS ---
    st.divider()
    st.subheader("Efficiency vs. Volume Analysis")
    
    view_option = st.selectbox(
        "Select Shot Category",
        ["Total PPP", "Three Point", "Mid Range", "Attack Basket"],
        key="scatter_view_select"
    )

    # Aggregate data based on view
    if view_option == "Total PPP":
        plot_df = conf_df.groupby('TEAM_NAME').agg({'TOTAL_POINTS_SCORED':'sum', 'PLAYS_RUN':'sum'}).reset_index()
        plot_df['PPP'] = plot_df['TOTAL_POINTS_SCORED'] / plot_df['PLAYS_RUN']
        
        # Benchmarks
        global_avg = (df_ato_ui['TOTAL_POINTS_SCORED'].sum() / df_ato_ui['PLAYS_RUN'].sum())
        tier_avg = (tier_df['TOTAL_POINTS_SCORED'].sum() / tier_df['PLAYS_RUN'].sum())
        conf_avg = (conf_df['TOTAL_POINTS_SCORED'].sum() / conf_df['PLAYS_RUN'].sum())
    else:
        plot_df = conf_df[conf_df['ATTEMPT_QUALITY'] == view_option].copy().rename(columns={'POINTS_PER_PLAY': 'PPP'})
        global_avg = df_ato_ui[df_ato_ui['ATTEMPT_QUALITY'] == view_option]['POINTS_PER_PLAY'].mean()
        tier_avg = tier_df[tier_df['ATTEMPT_QUALITY'] == view_option]['POINTS_PER_PLAY'].mean()
        conf_avg = conf_df[conf_df['ATTEMPT_QUALITY'] == view_option]['POINTS_PER_PLAY'].mean()

    fig_scatter = px.scatter(
        plot_df, x='PLAYS_RUN', y='PPP', color='TEAM_NAME', size='PLAYS_RUN',
        labels={'PLAYS_RUN': 'Plays Run', 'PPP': 'Points Per Play'},
        title=f"{view_option} Efficiency: {target_conf} vs Benchmarks",
        template="plotly_dark"
    )

    # Add the 3 Benchmark Lines
    fig_scatter.add_hline(y=global_avg, line_dash="dot", line_color="white", annotation_text="Global Avg")
    fig_scatter.add_hline(y=tier_avg, line_dash="dash", line_color="gray", annotation_text=f"{target_tier} Avg")
    fig_scatter.add_hline(y=conf_avg, line_dash="solid", line_color="#3B12F5", annotation_text="Conf Avg")
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 4. PLAY TYPE PROFILE (STACKED BAR) ---
    st.divider()
    st.subheader("ATO Play Type Distribution (%)")
    st.markdown("How teams utilize their timeouts (Shot Selection Profile)")

    # Calculate percentages for the stacked bar
    conf_df['Total Plays'] = conf_df.groupby('TEAM_NAME')['PLAYS_RUN'].transform('sum')
    conf_df['Pct of Plays'] = (conf_df['PLAYS_RUN'] / conf_df['Total Plays']) * 100

    fig_stack = px.bar(
        conf_df,
        x="TEAM_NAME",
        y="Pct of Plays",
        color="ATTEMPT_QUALITY",
        title=f"Play Type DNA: {target_conf} Conference",
        labels={"TEAM_NAME": "Team", "Pct of Plays": "Percentage of ATOs (%)", "ATTEMPT_QUALITY": "Shot Type"},
        barmode="relative", # Creates the 100% stack look
        template="plotly_dark"
    )
    fig_stack.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_stack, use_container_width=True)

    # --- 5. UI FRIENDLY DATA TABLE ---
    with st.expander("View Raw Comparative Data"):
        # Rename for UI
        ui_table = conf_df[['TEAM_NAME', 'ATTEMPT_QUALITY', 'PLAYS_RUN', 'TOTAL_POINTS_SCORED', 'POINTS_PER_PLAY']].copy()
        ui_table.columns = ['Team Name', 'Shot Type', 'Plays Run', 'Total Points', 'PPP']
        st.dataframe(ui_table.sort_values(['Team Name', 'PPP'], ascending=[True, False]), hide_index=True)  
