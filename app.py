import streamlit as st
import pandas
from src.clustering import cluster


def main():
    st.title("Crop Recommendations")

    N = st.number_input("Nitrogen Content (mg/kg)")
    P = st.number_input("Phosphorus Content (mg/kg)")
    K = st.number_input("Potassium Content (mg/kg)")
    temperature = st.number_input("Temperature (Celsius)")
    humidity = st.number_input("Relative Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")


    new_data = pandas.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    pred, corr, box_plots= st.tabs(["Prediction", "Correlation", "Box Plots"])

    with pred:
        sample_prediction, pie = cluster(new_data)

        st.write("Recommended Crop : ",sample_prediction)

        # Show pie chart of crop distributions for selected cluster
        st.pyplot(pie)

if __name__ == "__main__":
    main()