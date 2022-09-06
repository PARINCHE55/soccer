import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


def main():
    string = "Soccer Goal Predictor"
    st.set_page_config(page_title=string, page_icon="🤔")
    st.title("Soccer Goal Predictor Based on Weather Conditions")
    st.image(
        "https://img.freepik.com/premium-vector/soccer-player-with-ball-black-isolated-image-white-background-template-football-concept_619989-189.jpg",
        width=220,
        use_column_width=12

            )
    st.write('')
    st.write('')

    temperature = st.number_input('Temperature Value (°C)',min_value=4.00,max_value=42.00,step=0.1)
    humidity = st.number_input('Humidity (%)',min_value=5,max_value=100)
    pressure = st.number_input('Surface Pressure (hPa)',min_value=620,max_value=1043,step=10)
    cloudcover = st.number_input('Cloud Cover (%)',min_value=0.00,max_value=100.00,step=5.00)
    radiation = st.number_input('Solar Radiation (W/m²)',min_value=0.00,max_value=850.00,step=10.00)
    wind_speed = st.number_input('Wind Speed (km/h)',min_value=1.00,max_value=45.00,step=5.00)

    if st.button('Predict Goals'):
        query = np.array(
            [temperature, humidity, pressure, cloudcover, radiation, wind_speed])

        query = query.reshape(1, 6)
        st.title("Predicted Match Goals Are Between " + str(pipe.predict(query)[0]))


if __name__ == '__main__':
    main()

