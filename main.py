from util import *
from model import *

# Initialize streamlit app
page_title = "Breast Cancer Predictor"
page_icon = "👩‍⚕️"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide", initial_sidebar_state="expanded")

# Load the CSS file
with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# Configure sidebar for user input
input_data = configure_sidebar()

# Application title and description
st.title(page_title)
st.write(':blue[***Detect & Visualize Breast Cancer Risk with Precision 🎯***]')
st.write("This application allows users to input Cell Nuclei Measurement data and provides a prediction of breast "
         "cancer likelihood. It also visually represents the measurements in an intuitive radar chart 🕸️, making "
         "it easier to understand the crucial factors influencing the diagnosis. Fast, insightful, and "
         "life-saving 💡! ")

# Model selection
st.subheader('Select Machine Learning Model')
model_name = st.selectbox('Select the Model', ("Logistic Regression", "Decision Tree", "Random Forest", "Gaussian NB"),
                          label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric = train_model(model_name)
prediction, probability = model_predictions(input_data, model, scalar)

col1, col2 = st.columns([3, 1])

# Plot radar chart for input data and display model prediction
with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
with col2:
    display_prediction(prediction, probability)

# Display performance metrics
st.subheader('Performance Metrics')
display_performance_metrics(df_performance_metric)

# Display footer
display_footer()


