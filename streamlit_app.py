import json
import os

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit import session_state, container, pyplot
from ift6758.ift6758.data.front_end_data import FrontEndData

st.set_page_config(layout="wide")
st.title("NHL-game-APP")
url = ("http://serving-service:5000")

if "model_download" not in st.session_state:
    st.session_state.model_download = False
if "game_id" not in st.session_state:
    st.session_state.game_id = ""
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()



with st.sidebar:
    st.header("Choose a model")

    workspace = st.selectbox("Workspace",["Default"])
    model = st.selectbox("Choose a model",["RL-Distance","RL-Angle","Rl-DistanceAngle"])
    version = st.selectbox("Version",["1.0.0"])

    if st.button("Get Model"):
        st.write("Downloading the model")
        body = {"workspace": workspace, "model": model, "version": version}
        try:
           request = requests.post(
               f"{url}/download_registry_model",
               json=body
           )
           if request.status_code == 200:
               st.session_state.model_download = True
               st.success("Model Downloaded with success")
           else:
               st.warning("Model Download Failed")
        except Exception as e:
            st.error(e)


with (container()):
    st.header("Hockey Visualization data")
    st.session_state.game_id = st.text_input("Game ID",value="2022030411")
    button_enable = st.session_state.game_id.strip() == ""
    if not st.session_state.model_download:
        st.warning("No model selected")
        st.stop()
    if st.button("Ping game",disabled=button_enable):
            data = FrontEndData(session_state.game_id)
            raw_data = data.ping_game()

            if "Distance" in model:
                model_column = ["distanceToNet"]
            elif "Angle" in model:
                model_column = ["shotAngle"]
            elif "DistanceAngle" in model:
                model_column = ["distanceToNet","shotAngle"]

            try:
                request = requests.post(
                    f"{url}/predict",
                    json=json.loads(raw_data[model_column].to_json())
                )
                if request.status_code == 200:
                     df = raw_data
                     predictions = pd.read_json(request.json()["predictions"])
                     df['xg'] = predictions['expected_goal_prob'].values

                     st.subheader(f"Game {st.session_state.game_id} : {df['home'].iloc[0]} VS {df['away'].iloc[0]}")
                     st.write(f"Period {df['period'].iloc[0]}")

                     column1,column2 = st.columns(2)
                     with column1:
                         st.caption(f"{df['home'].iloc[0]} xG (actual)")
                         home_xg = df[df['team'] =='home']['xg'].sum()
                         st.markdown(f"### {home_xg: .1f} ({df['homeGoals'].iloc[0]})")
                         diff = home_xg-df['homeGoals'].iloc[0]
                         st.write(f"↓ {diff: .1f}" if diff<0 else f" ↑({df['homeGoals'].iloc[0]})")
                     with column2:
                         st.caption(f"{df['away'].iloc[0]} xG (actual)")
                         away_xg = df[df["team"] == "away"]["xg"].sum()
                         st.markdown(f"### {away_xg:.1f} {df['awayGoals'].iloc[0]}")
                         diff = away_xg - df['awayGoals'].iloc[0]
                         st.write(f"↓ {diff: .1f}" if diff < 0 else f" ↑({df['awayGoals'].iloc[0]})")

                     st.subheader("Data used for predictions")
                     st.session_state.data = df[['gameId', 'eventId', 'period', 'timeInPeriod','distanceToNet', 'shotAngle', 'is_goal', 'is_empty_net','team','xg']]
                     st.dataframe(st.session_state.data,use_container_width=True)


                     st.subheader("Bonus")
                     percentage_shoot_expected = st.session_state.data['xg'].sum()/st.session_state.data['is_goal'].sum()
                     col1,col2,col3 = st.columns(3)

                     col1.metric("Shots Total:",len(st.session_state.data))
                     col2.metric("Goals Total:",st.session_state.data['is_goal'].sum())
                     col3.metric("Expected Shooting:" ,f"{percentage_shoot_expected*100: .2f}%")

                     count_shots = pd.DataFrame()
                     count_shots['shot_types'] = st.session_state.data.apply(lambda x: "goal" if x.is_goal== 1 else ("shot on goal" if x.xg > 0 else "missed"),axis=1)

                     count_shots = count_shots['shot_types'].value_counts().reset_index()
                     count_shots.columns = ["shot_types","count"]


                     pie = px.pie(count_shots, names="shot_types",values='count',hole=0.6,color='shot_types',
                                  color_discrete_map={
                                      "Goal": "green",
                                      "Missed": "red",
                                      "Shot On Goal": "blue",
                                  })
                     pie.update_layout(title="Distribution of shots",showlegend=True)
                     st.plotly_chart(pie, use_container_width=True)
            except Exception as e:
                st.error(e)




