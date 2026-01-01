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
    st.title("⏱️ After Timeout (ATO) Efficiency")
    
    # --- 1. FILTERS (Nested in 3 columns) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        conf_tier = st.multiselect("Conference Tier", 
                                   options=sorted(df_ato['CONFERENCE_TYPE'].unique()),
                                   key="ato_tier")
    with col2:
        # Filter conference list based on tier if selected
        conf_options = df_ato[df_ato['CONFERENCE_TYPE'].isin(conf_tier)] if conf_tier else df_ato
        conf_list = st.multiselect("Conference", 
                                   options=sorted(conf_options['CONFERENCE'].unique()),
                                   key="ato_conf")
    with col3:
        # Filter team list based on previous selections
        team_options = conf_options[conf_options['CONFERENCE'].isin(conf_list)] if conf_list else conf_options
        selected_teams = st.multiselect("Select Teams", 
                                        options=sorted(team_options['TEAM_NAME'].unique()),
                                        key="ato_team_select")

    # --- 2. APPLY FILTERS ---
    # Start with full dataframe and narrow down
    filtered_ato = df_ato.copy()
    if conf_tier:
        filtered_ato = filtered_ato[filtered_ato['CONFERENCE_TYPE'].isin(conf_tier)]
    if conf_list:
        filtered_ato = filtered_ato[filtered_ato['CONFERENCE'].isin(conf_list)]
    if selected_teams:
        filtered_ato = filtered_ato[filtered_ato['TEAM_NAME'].isin(selected_teams)]

    # --- 3. KEY METRICS ---
    if not filtered_ato.empty:
        total_plays = filtered_ato['PLAYS_RUN'].sum()
        total_pts = filtered_ato['TOTAL_POINTS_SCORED'].sum()
        # Weighted average calculation for PPP
        avg_ppp = round(float(total_pts / total_plays), 3) if total_plays > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total ATO Plays", f"{total_plays:,}")
        m2.metric("Total Points", f"{total_pts:,}")
        m3.metric("Avg Points Per Play", avg_ppp)

        # --- 4. VISUALS ---
        st.divider()
        vis_col1, vis_col2 = st.columns([2, 1])

        with vis_col1:
            st.subheader("Efficiency by Attempt Quality")
            # Bar chart comparing PPP across different attempt types
            fig_bar = px.bar(
                filtered_ato.groupby('ATTEMPT_QUALITY', as_index=False).agg({'POINTS_PER_PLAY': 'mean'}),
                x='ATTEMPT_QUALITY',
                y='POINTS_PER_PLAY',
                color='ATTEMPT_QUALITY',
                text_auto='.3f',
                title="Avg Points Per Play by Category"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with vis_col2:
            st.subheader("Volume Distribution")
            # Pie chart showing how many plays fall into each quality category
            fig_pie = px.pie(
                filtered_ato, 
                values='PLAYS_RUN', 
                names='ATTEMPT_QUALITY',
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- 5. DATA TABLE ---
        st.subheader("Detailed Performance Breakdown")
        st.dataframe(
            filtered_ato[['TEAM_NAME', 'ATTEMPT_QUALITY', 'PLAYS_RUN', 'TOTAL_POINTS_SCORED', 'POINTS_PER_PLAY']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No data found for the selected filters. Please adjust your criteria.")
