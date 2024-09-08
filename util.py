import streamlit as st
import plotly.graph_objects as go
from model import *


# Configure the sidebar to get the Cell Nuclei measurements from user
def configure_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_breast_cancer_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    slider_labels_mean = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ]
    slider_labels_se = [
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
    ]
    slider_labels_worst = [
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict_mean = {}
    input_dict_se = {}
    input_dict_worst = {}

    st.sidebar.subheader('Mean Values:')
    with st.sidebar.container(border=True):
        for label, key in slider_labels_mean:
            input_dict_mean[key] = st.slider(label, min_value=float(0), max_value=float(data[key].max()),
                                             value=float(data[key].mean()))

    st.sidebar.subheader('Standard Error Values:')
    with st.sidebar.container(border=True):
        for label, key in slider_labels_se:
            input_dict_se[key] = st.slider(label, min_value=float(0), max_value=float(data[key].max()),
                                           value=float(data[key].mean()))

    st.sidebar.subheader('Worst Values:')
    with st.sidebar.container(border=True):
        for label, key in slider_labels_worst:
            input_dict_worst[key] = st.slider(label, min_value=float(0), max_value=float(data[key].max()),
                                              value=float(data[key].mean()))

    combined_input_dict = {**input_dict_mean, **input_dict_se, **input_dict_worst}
    return combined_input_dict


# Get the scaled values of breast cancer data. This is required to scale the radar chart
def get_scaled_values(input_dict):
    data = get_breast_cancer_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


# Plot radar chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


# Function to display the model prediction and probability
def display_prediction(prediction, probability):
    st.subheader("Cell Cluster Prediction")
    if prediction[0] == 0:
        st.write("Cell Cluster is: <span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
        st.write("Benign Probability: {:.3f}".format(probability[0][0]))
    else:
        st.write("Cell Cluster is: <span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
        st.write("Malignant Probability: {:.3f}".format(probability[0][1]))

    st.write('Disclaimer:')
    st.info('This application is for information purpose only and should not be considered as medical '
            'advice or a conclusive diagnosis. Always consult a qualified healthcare professional for an accurate '
            'diagnosis and personalized medical advice.')


# Function to display the footer
def display_performance_metrics(df_performance_metric):
    accuracy_col, f1_score_col, precision_col, recall_col, roc_auc_score_col = st.columns(5)
    with accuracy_col:
        with st.container(border=True):
            st.metric('*Accuracy*', value=df_performance_metric['Accuracy'].iloc[0])
    with f1_score_col:
        with st.container(border=True):
            st.metric('*F1 Score*', value=float(df_performance_metric['F1 Score'].iloc[0]))
    with precision_col:
        with st.container(border=True):
            st.metric('*Precision*', value=float(df_performance_metric['Precision'].iloc[0]))
    with recall_col:
        with st.container(border=True):
            st.metric('*Recall*', value=float(df_performance_metric['Recall'].iloc[0]))
    with roc_auc_score_col:
        with st.container(border=True):
            st.metric('*ROC AUC Score*', value=float(df_performance_metric['ROC AUC Score'].iloc[0]))


def display_footer():
    footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            text-align: center;
            color: grey;
            padding: 10px 0;
        }
        </style>
        <div class="footer">
            Made with ❤️ by <a href="mailto:zeeshan.altaf@92labs.ai">Zeeshan</a>.
            Source code <a href='https://github.com/mzeeshanaltaf/'>here</a>.</div> 
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
