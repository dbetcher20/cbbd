import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit as st
import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# 1. Page Config
st.set_page_config(page_title="Snowflake Sports Analytics", layout="wide")

# 2. Establish Snowflake Connection
def get_private_key():
    p_key_text = st.secrets["connections"]["snowflake"]["private_key_content"]
    passphrase = st.secrets["connections"]["snowflake"].get("private_key_passphrase")
    
    p_key_bytes = serialization.load_pem_private_key(
        p_key_text.encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend()
    ).private_key_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return p_key_bytes
# We pass the processed key directly into the connection
conn = st.connection("snowflake", private_key=get_private_key())

# 3. Cached Data Fetching
@st.cache_data(ttl=3600) # Caches results for 1 hour
def load_snowflake_data(query):
    # Executes the select statement and returns a Pandas DataFrame
    return conn.query(query)

st.title("🏆 Snowflake Sports Dashboard")

# 4. Execute Select Statement
# Replace 'SPORTS_VIEW' with the actual name of your Snowflake view
query = "select * from cbb_data.views.dim_team"

try:
    df = load_snowflake_data(query)

    # 5. Dynamic Sidebar Filters
    st.sidebar.header("Filters")
    # Using 'TEAM' as an example column from your Snowflake view
    teams = df['TEAM_NAME'].unique()
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams[:2])

    filtered_df = df[df['TEAM_NAME'].isin(selected_teams)]

    # 6. Visualizations
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.subheader("Performance Metric")
    #     # Replace 'PLAYER' and 'POINTS' with your actual column names
    #     fig = px.bar(filtered_df, x='PLAYER', y='POINTS', color='TEAM')
    #     st.plotly_chart(fig, use_container_width=True)

    # with col2:
    #     st.subheader("Metric Correlation")
    #     # Replace 'ASSISTS' and 'REBOUNDS' with your column names
    #     fig_scatter = px.scatter(filtered_df, x='ASSISTS', y='REBOUNDS', hover_name='PLAYER')
    #     st.plotly_chart(fig_scatter, use_container_width=True)

    # 7. Raw Data Table
    with st.expander("View Raw Data from Snowflake"):
        st.dataframe(filtered_df)

except Exception as e:
    st.error(f"Error connecting to Snowflake: {e}")