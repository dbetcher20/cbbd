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

    # --- 1. FILTERS ---
    # To compare a team vs their conference, we need to know which conference they are in
    team_list = sorted(df_ato['TEAM_NAME'].unique())
    selected_team = st.selectbox("Select Primary Team", team_list, key="ato_main_team")
    
    # Identify the conference for the selected team
    target_conf = df_ato[df_ato['TEAM_NAME'] == selected_team]['CONFERENCE'].iloc[0]
    
    # Create the comparison dataset (All teams in that specific conference)
    conf_df = df_ato[df_ato['CONFERENCE'] == target_conf].copy()
    # Convert Decimals to floats for Plotly/Math
    conf_df['POINTS_PER_PLAY'] = conf_df['POINTS_PER_PLAY'].astype(float)
    conf_df['PLAYS_RUN'] = conf_df['PLAYS_RUN'].astype(float)

    # --- 2. RANKED SORTED ORDER (BAR CHART) ---
    st.subheader(f"Ranked ATO Efficiency: {target_conf} Conference")
    
    # Sort teams by efficiency
    ranked_df = conf_df.groupby('TEAM_NAME')['POINTS_PER_PLAY'].mean().reset_index()
    ranked_df = ranked_df.sort_values('POINTS_PER_PLAY', ascending=False)
    
    # Highlight the selected team in a different color
    ranked_df['Color'] = ranked_df['TEAM_NAME'].apply(lambda x: '#FF4B4B' if x == selected_team else '#31333F')

    fig_rank = px.bar(
        ranked_df,
        x='POINTS_PER_PLAY',
        y='TEAM_NAME',
        orientation='h',
        text_auto='.3f',
        color='Color',
        color_discrete_map="identity",
        title=f"Who is most efficient in the {target_conf}?"
    )
    fig_rank.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    # --- 3. SCATTERPLOT (VOLUME VS EFFICIENCY) ---
    st.divider()
    st.subheader("Efficiency vs. Volume")
    
    fig_scatter = px.scatter(
        conf_df,
        x='PLAYS_RUN',
        y='POINTS_PER_PLAY',
        color='ATTEMPT_QUALITY',
        hover_name='TEAM_NAME',
        size='TOTAL_POINTS_SCORED',
        labels={'PLAYS_RUN': 'Volume (Plays Run)', 'POINTS_PER_PLAY': 'Efficiency (PPP)'},
        title=f"ATO Comparison for all {target_conf} Teams"
    )
    
    # Add a horizontal line for the conference average efficiency
    avg_conf_ppp = conf_df['POINTS_PER_PLAY'].mean()
    fig_scatter.add_hline(y=avg_conf_ppp, line_dash="dash", annotation_text="Conf Avg")
    
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 4. DATA SUMMARY ---
    with st.expander("View Comparative Metrics"):
        st.dataframe(conf_df.sort_values(['TEAM_NAME', 'POINTS_PER_PLAY'], ascending=[True, False]))
        
