import json, requests
import streamlit as st
import pandas as pd
from live_game_event import LiveGameClient
import os
import plotly.express as px
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Milestone 3 - NHL xG Dashboard",
                   layout="wide")

if "game_id" not in st.session_state:
    st.session_state['game_id'] = ""

if "model_loaded" not in st.session_state:
    st.session_state['model_loaded'] = False

if "shot_data" not in st.session_state:
    st.session_state['shot_data'] = pd.DataFrame()

selected_workspace = st.sidebar.selectbox("Workspace", ["Default"])
user_model_choice = st.sidebar.selectbox("Model", ["Distance",
                                                   "Distance and Angle"])
selected_version = st.sidebar.selectbox("Version", ["1.0"])

if user_model_choice == 'Distance':
    selected_model = 'Distance'
elif user_model_choice == 'Distance and Angle':
    selected_model = 'DistanceAngle'

if st.sidebar.button("Load Model"):

    payload = {"workspace": selected_workspace,
               "model": selected_model,
               "version": selected_version}

    with st.spinner(f"Downloading model: {selected_model}-based Logistic Regression"):
        try:
            r = requests.post(f"{BACKEND_URL}/download_registry_model", json=payload)

            if r.status_code == 200:
                st.session_state['model_loaded'] = True
                st.sidebar.success(r.json()['message'])
                st.session_state['shot_data'] = pd.DataFrame()

            else:
                st.sidebar.error(f"An Error: {r.json().get('message')}")

        except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")

st.title("Hockey Visualization App")
game_id_input = st.text_input("Game ID:", value=2023020204)

if st.button("Ping game"):

    if not st.session_state['model_loaded']:
        st.error("Load a model in the sidebar before pinging a game!")
        st.stop()

    if "client" not in st.session_state or st.session_state['game_id'] != game_id_input:
        st.session_state['client'] = LiveGameClient(game_id=game_id_input)
        st.session_state['game_id'] = game_id_input
        st.session_state['all_shots'] = pd.DataFrame()

    client = st.session_state['client']

    try:
        raw_data = client.get_data_from_id()
        data = pd.json_normalize(raw_data)

        away_name_last = data['awayTeam.commonName.default'].iloc[0]
        home_name_last = data['homeTeam.commonName.default'].iloc[0]
        away_score = data['awayTeam.score'].iloc[0]
        home_score = data['homeTeam.score'].iloc[0]
        curr_period = data['periodDescriptor.number'].iloc[0]
        remaining_time = data['clock.timeRemaining'].iloc[0]

        st.subheader(f"Game {game_id_input}: {home_name_last} vs {away_name_last}")
        st.write(f"Period {curr_period} - {remaining_time} left")

    except Exception:
        st.warning("Could not fetch game metadata (the game might not have started).")

    new_shots = client.process_data_from_id()

    if not new_shots.empty:

        if selected_model == "Distance":
            feature_cols = ["distanceToNet"]
        else:
            feature_cols = ["distanceToNet", "shotAngle"]

        model_input = new_shots[feature_cols].reset_index(drop=True)

        try:
            r = requests.post(f"{BACKEND_URL}/predict",
                              json=json.loads(model_input.to_json(orient="columns")))

            preds = pd.read_json(r.json()["predictions"]).reset_index(drop=True)
            new_shots['xg'] = preds['expected_goal_prob'].values
            st.session_state['all_shots'] = pd.concat([st.session_state['all_shots'], new_shots], ignore_index=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            pass

    if st.session_state['all_shots'].empty:
        st.info("No shots recorded yet.")

    else:
        xg_home = st.session_state['all_shots'].loc[st.session_state['all_shots'].team == "home", "xg"].sum()
        xg_away = st.session_state['all_shots'].loc[st.session_state['all_shots'].team == "away", "xg"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Home xG (actual)",
                      f"{round(xg_home, 2)} ({home_score})",
                      round(home_score - xg_home, 2))
        with col2:
            st.metric(f"Away xG (actual)",
                      f"{round(xg_away, 2)} ({away_score})",
                      round(away_score - xg_away, 2))

        st.subheader("All shot events")
        st.dataframe(st.session_state['all_shots'])

        st.subheader("Bonus")
        percentage_shoot_expected = st.session_state['all_shots']['xg'].sum() / st.session_state['all_shots']['is_goal'].sum()
        col1, col2, col3 = st.columns(3)

        col1.metric("Shots Total:", len(st.session_state['all_shots']))
        col2.metric("Goals Total:", st.session_state['all_shots']['is_goal'].sum())
        col3.metric("Expected Shooting:", f"{percentage_shoot_expected * 100: .2f}%")

        count_shots = pd.DataFrame()
        count_shots['shot_types'] = st.session_state['all_shots'].apply(
            lambda x: "goal" if x.is_goal == 1 else ("shot on goal" if x.xg > 0 else "missed"), axis=1)

        count_shots = count_shots['shot_types'].value_counts().reset_index()
        count_shots.columns = ["shot_types", "count"]

        pie = px.pie(count_shots, names="shot_types", values='count', hole=0.6, color='shot_types',
                     color_discrete_map={
                         "Goal": "green",
                         "Missed": "red",
                         "Shot On Goal": "blue",
                     })
        pie.update_layout(title="Distribution of shots", showlegend=True)
        st.plotly_chart(pie, use_container_width=True)
