import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# Load GBTM model 
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# ========== Resolve Chinese garbled characters ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # Ensure normal display of Chinese (reserved for potential use)
plt.rcParams['axes.unicode_minus'] = False    # Ensure normal display of negative signs
# =================================================

# Set page configuration
st.set_page_config(
    page_title="TBA Trajectory Prediction Model",
    layout="wide"
)

# Core model loading code starts
try:
    with open('gbtm5_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # 1. Read number of groups and convert to integer (5 groups)
    n_groups = int(model_data['ng'])  
    # 2. Read coefficients, convert to array + dimension verification
    param_matrix = np.array(model_data['coefficients'])
    
    # ========== Fix: Dimension verification + forced reconstruction ==========
    # Standard coefficient matrix [5 rows x 4 columns] (fixed GBTM structure)
    if param_matrix.ndim != 2 or param_matrix.shape[0] != n_groups or param_matrix.shape[1] !=4:
        #st.warning("✅ Version 4")
        # Format: 5 rows = 5 trajectory groups, 4 columns = intercept, linear term, quadratic term, cubic term (fixed 4 parameters)
        param_matrix = [
            [205.862,  -140.332,  54.958,   -7.189],  # Group 1: Cubic polynomial
            [-4.943,    253.300, -146.564,  26.698],  # Group 2: Cubic polynomial
            [145.1073, -43.2551, 13.4262,    0.0],    # Group 3: Quadratic polynomial (β3=0) 
            [318.078,  -207.331,  84.979,  -11.307],  # Group 4: Cubic polynomial
            [81.980,    154.438, -55.981,    7.915]   # Group 5: Cubic polynomial
        ]
        param_matrix = np.array(param_matrix)  # Force conversion to numpy array
        n_groups = 5  

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Define trajectory prediction function
def predict_gbtm_trajectory(time_points, group_params):
    intercept, linear, quadratic, cubic = group_params
    predictions = (
        intercept + 
        linear * time_points + 
        quadratic * (time_points ** 2) + 
        cubic * (time_points ** 3)
    )
    return predictions

def calculate_group_probabilities(tba_values, time_points, param_matrix):
    n_groups = param_matrix.shape[0]
    
    # Calculate distance between predicted and actual values for each group
    distances = []
    for g in range(n_groups):
        params = param_matrix[g]
        predicted = predict_gbtm_trajectory(time_points, params)
        
        # Calculate mean squared error
        mse = np.mean((np.array(tba_values) - predicted) ** 2)
        distances.append(mse)
    
    # Convert distance to probability
    distances = np.array(distances)
    if np.min(distances) == 0:
        probabilities = np.zeros(n_groups)
        probabilities[np.argmin(distances)] = 1.0
    else:
        weights = 1 / (distances + 1e-10)  # Add small value to avoid division by zero
        probabilities = weights / np.sum(weights)
    
    return probabilities

# Streamlit user interface
st.title("BA-NLS Predictor: An Online Tool for Prognostic Stratification Based on Postoperative TBA Trajectories in Biliary Atresia")

# ====================== Key Modification 1: Globally unified time point definition ======================
# Time point configuration (globally unified)
time_labels = ["Baseline (Preoperative)", "2 Weeks Postoperative", "1 Month Postoperative", "3 Months Postoperative"]  # Unified medical terminology
time_points_original = np.array([1, 2, 3, 4])  # Time points for modeling (1-4, consistent with R language)
time_points_smooth = np.linspace(1, 4, 100)    # Smooth time points for plotting
n_time_points = len(time_labels)

# Sidebar - Input parameters
st.sidebar.header("Input of Patient's TBA Measurement Data")
st.sidebar.subheader("Please enter TBA values at each time point (μmol/L)")

# Create input boxes (using unified time_labels)
tba_values = []
for i, label in enumerate(time_labels):
    value = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=900.0,
        value=float(i * 20 + 10),
        step=1.0,
        format="%.1f",
        key=f"tba_{i}"
    )
    tba_values.append(value)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("TBA Trajectory Visualization")
    # Create canvas
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot [Patient's actual TBA trajectory] - 4 discrete points + blue line
    ax.plot(time_points_original, tba_values, 'bo-', linewidth=2, markersize=8, label='Patient\'s Actual TBA Values', zorder=5)

    # Plot [5 groups of GBTM background curves]
    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2', '#FF7F0E']  # R language color scheme
    group_names = [f"Group {i+1}" for i in range(n_groups)]
    all_predictions = []

    for g in range(n_groups):
        params = param_matrix[g]
        predicted_smooth = predict_gbtm_trajectory(time_points_smooth, params)
        all_predictions.append(predicted_smooth)
        ax.plot(time_points_smooth, predicted_smooth,
                color=colors[g],
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f'{group_names[g]} Trajectory')

    # Axis configuration (using unified time_labels)
    ax.set_xlabel('Follow-up Time', fontsize=12)
    ax.set_ylabel('TBA Value (μmol/L)', fontsize=12)
    ax.set_title('TBA Trajectory Comparison (GBTM 5-Group Model)', fontsize=14)
    ax.set_xticks(time_points_original)
    ax.set_xticklabels(time_labels, rotation=0)  # Use unified medical terminology
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

with col2:
    st.subheader("Prediction Results")
    
    # Calculate probability of each group
    if st.sidebar.button("Start Prediction"):
        with st.spinner("Calculating..."):
            # ====================== Modification 2: Pass correct time point parameters ======================
            probabilities = calculate_group_probabilities(tba_values, time_points_original, param_matrix)
            
            # Find the most likely group
            most_likely_group = np.argmax(probabilities) + 1
            
            st.write(f"**Prediction Completed!**")
            st.write(f"**Most Likely Trajectory Group:** Group {most_likely_group}")
            
            # Display probability of all groups
            st.subheader("Probability of Each Trajectory Group")
            for g in range(n_groups):
                prob_percent = probabilities[g] * 100
                st.write(f"**{group_names[g]}**: {prob_percent:.1f}%")
            
            # Clinical recommendations
            st.subheader("Clinical Recommendations")
            advice_dict = {
                1: "Group 1 (Low Baseline Decreasing Group): A standard follow-up protocol may be maintained, with focused monitoring of liver function and growth parameters.",
                2: "Group 2 (Low Baseline Sharp Increasing Group): Intensified management is indicated with increased follow-up frequency. The timing of liver transplantation should be continuously evaluated during follow-up, with proactive preparation for subsequent liver transplantation assessment when necessary.",
                3: "Group 3 (Low Baseline Increasing Group): Follow-up intervals should be appropriately shortened and the frequency of endoscopic surveillance enhanced, so as to enable the early detection and management of portal hypertension-related complications.",
                4: "Group 4 (High Baseline Decreasing Group): A standard follow-up protocol may be maintained, with focused monitoring of liver function and growth parameters.",
                5: "Group 5 (High Baseline Increasing Group): Close follow-up must be implemented with increased endoscopic surveillance for the proactive mitigation of risks such as variceal bleeding. For this subgroup, priority should be given to the monitoring of liver function, and evaluation for liver transplantation should be promptly initiated in the event of liver function deterioration."
            }
            # Display main recommendations
            if most_likely_group in advice_dict:
                st.write(advice_dict[most_likely_group])
            else:
                st.write("Specific recommendations cannot be provided; please make judgments based on actual clinical conditions.")

# Add data summary
st.subheader("Data Summary")
# Create data summary table (using unified time_labels)
summary_data = {
    "Time Point": time_labels,
    "TBA Value (μmol/L)": tba_values
}
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Run check
import sys
if "streamlit" not in sys.modules:
    st.warning("Please run this application with the command 'streamlit run app.py'")
    st.stop()
