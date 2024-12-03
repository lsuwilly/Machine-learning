import streamlit as st
import pandas
from src.clustering import find_cluster
from src.box_plots import make_box_plots
from src.correlation_grid import get_correlation_grid


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
        sample_prediction, pie = find_cluster(new_data)

        st.markdown(f"### Recommended Crop for these growing conditions : {sample_prediction}")
        st.markdown("")

        # Show pie chart of crop distributions for selected cluster
        st.pyplot(pie)

    with corr:
        grid = get_correlation_grid()

        st.pyplot(grid)
    
    with box_plots:
        plots = make_box_plots()

        for plot in plots:
            st.pyplot(plot)

if __name__ == "__main__":
    main()