import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# ---------------------------------------
# PAGE CONFIG MUST BE FIRST!
# ---------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# LOAD MODEL - TRY ALL AVAILABLE FORMATS
# ---------------------------------------
def load_model():
    import tensorflow as tf
    from keras.layers import Input, TFSMLayer
    from keras.models import Model
    import os

    tf_version = tf.__version__
    model_path = "multitask_loan_model_tf"

    if not os.path.isdir(model_path):
        st.sidebar.error("❌ SavedModel folder not found")
        return None, None, tf_version

    try:
        # Load SavedModel as inference-only layer
        tfsm_layer = TFSMLayer(
            model_path,
            call_endpoint="serving_default"
        )

        # Wrap SavedModel inside a Keras Model
        inputs = Input(shape=(None,), dtype=tf.float32)
        outputs = tfsm_layer(inputs)

        model = Model(inputs=inputs, outputs=outputs)

        st.sidebar.success("✅ SavedModel loaded via TFSMLayer (Keras 3 safe)")
        return model, model_path, tf_version

    except Exception as e:
        st.sidebar.error(f"❌ Failed to load SavedModel: {e}")
        return None, None, tf_version

# Load the model
model, model_source, tf_version = load_model()

# ---------------------------------------
# Load other resources
# ---------------------------------------
@st.cache_data
def load_resources():
    """Load all data files"""
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")
        st.sidebar.success(f"✅ Loaded {len(X_test)} test samples")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
    
    # Load SHAP explanations
    shap_data = {}
    for name in ['pd', 'ttd', 'lgd']:
        try:
            with open(f"shap_{name}.pkl", "rb") as f:
                shap_data[name] = pickle.load(f)
        except:
            shap_data[name] = None
    
    # Load feature names
    try:
        feature_names = np.load("feature_names.npy", allow_pickle=True)
    except:
        feature_names = X_test.columns.tolist()
    
    return X_test, y_test, shap_data, feature_names

# Load data
X_test, y_test, shap_data, feature_names = load_resources()

# ---------------------------------------
# APP UI
# ---------------------------------------
st.title("🏦 Loan Default Risk Analysis Dashboard")
st.markdown(f"""
**Multi-Task Neural Network Predictions**  
*TensorFlow {tf_version} | Model: {model_source if model_source else 'Not loaded'}*

- **📊 Probability of Default (PD)** - Likelihood of borrower default
- **⏰ Time to Default (TTD)** - Expected months until default  
- **💰 Loss Given Default (LGD)** - Percentage loss if default occurs
""")

# ---------------------------------------
# Sample Selection
# ---------------------------------------
st.sidebar.header("🔧 Configuration")
idx = st.sidebar.slider("Select sample:", 0, len(X_test)-1, 0)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dataset Info:**
- Samples: {len(X_test):,}
- Features: {len(feature_names)}
- Current: #{idx}
- SHAP: {'✅ Loaded' if shap_data['pd'] is not None else '❌ Not loaded'}
- Model: {model_source if model_source else '❌ Not loaded'}
- TF Version: {tf_version}
""")

# ---------------------------------------
# Get predictions from model
# ---------------------------------------
def get_model_predictions(sample, model_obj):
    try:
        import tensorflow as tf
        x = tf.convert_to_tensor(sample.values.astype(np.float32))

        outputs = model_obj(x)

        # 🔒 EXACT keys (confirmed by inspection)
        pd_pred  = float(outputs["output_0"].numpy()[0][0])
        ttd_pred = float(outputs["output_1"].numpy()[0][0])
        lgd_pred = float(outputs["output_2"].numpy()[0][0])

        # Safety bounds
        pd_pred  = float(np.clip(pd_pred, 0.01, 0.99))
        ttd_pred = float(max(1.0, ttd_pred))
        lgd_pred = float(np.clip(lgd_pred, 0.01, 0.99))

        return pd_pred, ttd_pred, lgd_pred

    except Exception as e:
        st.sidebar.error(f"Prediction failed: {repr(e)}")
        return 0.25, 18.0, 0.45

# Get sample and predictions
sample = X_test.iloc[idx:idx+1]
pd_pred, ttd_pred, lgd_pred = get_model_predictions(sample, model)

# ---------------------------------------
# Display Predictions
# ---------------------------------------
st.header("📈 Model Predictions")

col1, col2, col3 = st.columns(3)
# Decide if model predicts default
default_predicted = pd_pred >= 0.5

# Add custom CSS for equal height
st.markdown("""
<style>
.equal-height-container {
    height: 280px;
    display: grid;
    grid-template-rows: auto 1fr auto;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid;
}
</style>
""", unsafe_allow_html=True)

with col1:
    pd_color = "#ff4d4d" if pd_pred > 0.7 else "#ffa64d" if pd_pred > 0.3 else "#4dff4d"
    st.markdown(f"""
    <div class="equal-height-container" style='background-color:{pd_color}20; border-color:{pd_color}'>
        <h3 style='margin-top:0; color:{pd_color}'>Probability of Default</h3>
        <div style='display:flex; align-items:center; justify-content:center;'>
            <h1 style='color:{pd_color}; margin:0'>{pd_pred*100:.1f}%</h1>
        </div>
        <p style='margin-bottom:0'>Risk: {'High' if pd_pred > 0.7 else 'Medium' if pd_pred > 0.3 else 'Low'}</p>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(pd_pred))

with col2:
    ttd_color = "#ff4d4d" if ttd_pred < 12 else "#4dff4d" if ttd_pred > 24 else "#ffa64d"
    
    if default_predicted:
        st.markdown(f"""
        <div class="equal-height-container" style='background-color:{ttd_color}20; border-color:{ttd_color}'>
            <h3 style='margin-top:0; color:{ttd_color}'>Time to Default</h3>
            <div style='display:flex; align-items:center; justify-content:center;'>
                <h1 style='color:{ttd_color}; margin:0'>{ttd_pred:.1f} months</h1>
            </div>
            <p style='margin-bottom:0'>Timeline: {'Short' if ttd_pred < 12 else 'Long' if ttd_pred > 24 else 'Medium'}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(ttd_pred))
    else:
        st.markdown(f"""
        <div class="equal-height-container" style='background-color:{ttd_color}20; border-color:{ttd_color}'>
            <h3 style='margin-top:0; color:{ttd_color}'>Time to Default</h3>
            <div style='display:flex; align-items:center; justify-content:center;'>
                <h2 style='color:{ttd_color}; margin:0'>Not Applicable</h2>
            </div>
            <p style='margin-bottom:0'>Timeline: No TTD predicted</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    lgd_color = "#ff4d4d" if lgd_pred > 0.6 else "#ffa64d" if lgd_pred > 0.4 else "#4dff4d"
    
    if default_predicted:
        st.markdown(f"""
        <div class="equal-height-container" style='background-color:{lgd_color}20; border-color:{lgd_color}'>
            <h3 style='margin-top:0; color:{lgd_color}'>Loss Given Default</h3>
            <div style='display:flex; align-items:center; justify-content:center;'>
                <h1 style='color:{lgd_color}; margin:0'>{lgd_pred*100:.1f}%</h1>
            </div>
            <p style='margin-bottom:0'>Severity: {'High' if lgd_pred > 0.6 else 'Medium' if lgd_pred > 0.4 else 'Low'}</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(lgd_pred))
    else:
        st.markdown(f"""
        <div class="equal-height-container" style='background-color:{lgd_color}20; border-color:{lgd_color}'>
            <h3 style='margin-top:0; color:{lgd_color}'>Loss Given Default</h3>
            <div style='display:flex; align-items:center; justify-content:center;'>
                <h2 style='color:{lgd_color}; margin:0'>Not Applicable</h2>
            </div>
            <p style='margin-bottom:0'>Severity: No LGD predicted</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------
# Actual Values
# ---------------------------------------
if not y_test.empty:
    st.header("📊 Actual Test Outcomes")
    
    actual_cols = st.columns(3)
    
    # Get actual default status
    actual_default_occurred = False
    if "default" in y_test.columns:
        actual_pd = y_test.iloc[idx]["default"]
        actual_default_occurred = actual_pd == 1
    
    with actual_cols[0]:
        if "default" in y_test.columns:
            st.metric("Actual Default", "✅ Yes" if actual_default_occurred else "❌ No")
        else:
            st.metric("Actual Default", "N/A")
    
    with actual_cols[1]:
        if "time_to_default" in y_test.columns:
            if actual_default_occurred:
                actual_ttd = y_test.iloc[idx]["time_to_default"]
                st.metric("Actual TTD", f"{actual_ttd:.1f} months")
            else:
                st.metric("Actual TTD", "Not Applicable")
        else:
            st.metric("Actual TTD", "N/A")
    
    with actual_cols[2]:
        if "lgd" in y_test.columns:
            if actual_default_occurred:
                actual_lgd = y_test.iloc[idx]["lgd"]
                st.metric("Actual LGD", f"{actual_lgd*100:.1f}%")
            else:
                st.metric("Actual LGD", "Not Applicable")
        else:
            st.metric("Actual LGD", "N/A")

# ---------------------------------------
# SHAP Helper Functions (DEFINED BEFORE USE)
# ---------------------------------------
def compute_shap_for_single_sample(sample, model_obj, background_samples=50):
    """Compute SHAP values for a single sample"""
    import shap
    
    if model_obj is None:
        return None
    
    try:
        # Use first n samples as background (or you could use k-means)
        background = X_test.iloc[:background_samples].values
        
        # Create explainer with proper model output
        def model_predict_wrapper(x):
            import tensorflow as tf

            x = tf.convert_to_tensor(x.astype(np.float32))
            outputs = model(x)

            pd  = outputs["output_0"].numpy()
            ttd = outputs["output_1"].numpy()
            lgd = outputs["output_2"].numpy()

            return np.column_stack([pd, ttd, lgd])
        
        # Create explainer
        explainer = shap.Explainer(model_predict_wrapper, background)
        
        # Compute SHAP for this single sample
        sample_array = sample.values.astype(np.float32)
        shap_values = explainer(sample_array)
        
        return shap_values
        
    except Exception as e:
        st.sidebar.warning(f"SHAP computation failed: {e}")
        return None

def plot_computed_shap(shap_values, title, idx):
    """Plot computed SHAP values for a single sample"""
    if shap_values is None:
        return
    
    # SHAP values shape: (1, n_features, 3) for 3 outputs
    # Extract values for PD (output 0), TTD (output 1), LGD (output 2)
    shap_pd = shap_values.values[0, :, 0]
    shap_ttd = shap_values.values[0, :, 1]
    shap_lgd = shap_values.values[0, :, 2]
    
    # Create tabs for each output
    tab1, tab2, tab3 = st.tabs(["📊 PD", "⏰ TTD", "💰 LGD"])
    
    with tab1:
        plot_single_shap_plot(shap_pd, f"{title} - Probability of Default")
    
    with tab2:
        plot_single_shap_plot(shap_ttd, f"{title} - Time to Default")
    
    with tab3:
        plot_single_shap_plot(shap_lgd, f"{title} - Loss Given Default")

def plot_single_shap_plot(shap_values_1d, title):
    """Plot a single SHAP values array"""
    # Get top 10 features
    top_indices = np.argsort(np.abs(shap_values_1d))[-10:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    features_top = [feature_names[i] for i in top_indices]
    values_top = [shap_values_1d[i] for i in top_indices]
    
    # Create bars
    bars = ax.barh(features_top, values_top, alpha=0.7)
    
    # Color bars and add labels
    for bar, value in zip(bars, values_top):
        bar_width = bar.get_width()
        bar_y = bar.get_y()
        bar_h = bar.get_height()

        # Positive SHAP value (right side)
        if bar_width >= 0:
            bar.set_color('#4CAF50')
            text_x = bar_width * 0.85   # keep text inside right bar
            ha = 'right'
        # Negative SHAP value (left side)
        else:
            bar.set_color('#F44336')
            text_x = bar_width * 0.85   # negative * 0.85 stays inside bar
            ha = 'left'

        # Ensure label never crosses zero (important fix)
        if bar_width > 0:
            text_x = max(text_x, bar_width * 0.2)
        else:
            text_x = min(text_x, bar_width * 0.2)

        ax.text(
            text_x,
            bar_y + bar_h / 2,
            f'{value:.4f}',
            va='center',
            ha=ha,
            color='black',
            fontweight='bold',
            fontsize=9,
            clip_on=True
        )
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("SHAP Value (Positive = Increases Prediction)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=1)
    
    st.pyplot(fig)
    plt.close(fig)

def plot_shap_for_target(shap_exp, title, idx):
    """Plot SHAP values from pre-computed data (fallback)"""
    if shap_exp is None:
        st.info(f"No SHAP data for {title}")
        return
    
    # Check if index is within bounds of SHAP data
    shap_size = shap_exp.values.shape[0]
    if idx >= shap_size:
        safe_idx = idx % shap_size
        st.info(f"Pre-computed SHAP data only has {shap_size} samples. Using sample #{safe_idx}.")
    else:
        safe_idx = idx
    
    shap_values = shap_exp.values[safe_idx]
    if shap_values.ndim == 2:
        shap_values = shap_values[:, 0]
    
    plot_single_shap_plot(shap_values, f"{title} (Sample #{safe_idx})")

# ---------------------------------------
# SHAP Explanations - IMPROVED VERSION
# ---------------------------------------
st.header("🎯 Feature Impact Analysis (SHAP)")

# Create columns to place radio buttons and compute button side by side
col_shap1, col_shap2 = st.columns([3, 1])

with col_shap1:
    # Add a toggle for computation method
    shap_method = st.radio(
        "SHAP Computation Method:",
        ["Use Pre-computed SHAP (Fast)", "Compute SHAP On-the-fly (Accurate)"],
        horizontal=True
    )

with col_shap2:
    # Add compute button (only show when on-the-fly is selected)
    if shap_method == "Compute SHAP On-the-fly (Accurate)":
        compute_shap = st.button("🔬 Compute SHAP", type="primary", use_container_width=True)

# Handle SHAP display based on selection
if shap_method == "Compute SHAP On-the-fly (Accurate)":
    # On-the-fly computation
    if model is None:
        st.warning("Model not loaded. Cannot compute SHAP.")
    elif 'compute_shap' in locals() and compute_shap:
        # Only compute when button is clicked
        with st.spinner("Computing SHAP values... (This may take 10-30 seconds)"):
            shap_values = compute_shap_for_single_sample(sample, model)
            
            if shap_values is not None:
                st.success("✅ SHAP computation completed!")
                plot_computed_shap(shap_values, "Real-time SHAP Analysis", idx)
            else:
                st.error("Failed to compute SHAP values.")
    else:
        # Show instruction to click button
        st.info("Click the '🔬 Compute SHAP' button above to compute SHAP values for the current sample.")
    
    # Still show pre-computed global SHAP if available
    if shap_data['pd'] is not None:
        st.subheader("🌍 Pre-computed Global Feature Importance")
        global_tabs = st.tabs(["PD Global", "TTD Global", "LGD Global"])
        
        with global_tabs[0]:
            if shap_data['pd'] is not None:
                shap_values = shap_data['pd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "PD - Global Feature Importance")
        
        with global_tabs[1]:
            if shap_data['ttd'] is not None:
                shap_values = shap_data['ttd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "TTD - Global Feature Importance")
        
        with global_tabs[2]:
            if shap_data['lgd'] is not None:
                shap_values = shap_data['lgd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "LGD - Global Feature Importance")

else:
    # Use pre-computed SHAP
    if shap_data['pd'] is not None:
        # Create tabs for each prediction
        tab1, tab2, tab3 = st.tabs(["📊 PD", "⏰ TTD", "💰 LGD"])
        
        with tab1:
            plot_shap_for_target(shap_data['pd'], "Probability of Default", idx)
        
        with tab2:
            plot_shap_for_target(shap_data['ttd'], "Time to Default", idx)
        
        with tab3:
            plot_shap_for_target(shap_data['lgd'], "Loss Given Default", idx)
        
        # Global SHAP summary
        st.subheader("🌍 Global Feature Importance")
        
        global_tabs = st.tabs(["PD Global", "TTD Global", "LGD Global"])
        
        with global_tabs[0]:
            if shap_data['pd'] is not None:
                shap_values = shap_data['pd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "PD - Global Feature Importance")
        
        with global_tabs[1]:
            if shap_data['ttd'] is not None:
                shap_values = shap_data['ttd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "TTD - Global Feature Importance")
        
        with global_tabs[2]:
            if shap_data['lgd'] is not None:
                shap_values = shap_data['lgd'].values
                if shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 0]
                n_samples = min(500, shap_values.shape[0])
                mean_abs_shap = np.mean(np.abs(shap_values[:n_samples]), axis=0)
                plot_single_shap_plot(mean_abs_shap, "LGD - Global Feature Importance")
    else:
        st.warning("No pre-computed SHAP data available.")
        
# ---------------------------------------
# Feature Values
# ---------------------------------------
st.header("📋 Feature Values for Selected Sample")

with st.expander("View feature values"):
    # Show top 10 most varying features
    feature_std = X_test.std().sort_values(ascending=False)
    top_features = feature_std.head(10).index.tolist()
    
    feature_data = []
    for feat in top_features:
        feature_data.append({
            "Feature": feat,
            "Value": X_test.iloc[idx][feat],
            "Mean": X_test[feat].mean(),
            "Std": X_test[feat].std(),
            "Z-Score": (X_test.iloc[idx][feat] - X_test[feat].mean()) / X_test[feat].std()
        })
    
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True)

# ---------------------------------------
# Model Information
# ---------------------------------------
if model is not None:
    with st.expander("🔧 Model Details"):
        st.write(f"**Loaded from:** {model_source}")
        st.write(f"**TensorFlow:** {tf_version}")
        
        if hasattr(model, 'input_shape'):
            st.write(f"**Input Shape:** {model.input_shape}")
        
        if hasattr(model, "output"):
            st.write("**Outputs:**")

            if isinstance(model.output, dict):
                for name, tensor in model.output.items():
                    st.write(f"  - {name}: {tensor.shape}")

            elif isinstance(model.output, list):
                for i, tensor in enumerate(model.output):
                    st.write(f"  - Output {i+1}: {tensor.shape}")

            else:
                st.write(f"  - Output: {model.output.shape}")
        
        # Show model type
        model_type = type(model).__name__
        st.write(f"**Model Type:** {model_type}")

# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Credit Risk Analysis Dashboard</strong></p>
    <p style='font-size: 0.9em; color: #666;'>
        Multi-task neural network for PD, TTD, and LGD prediction<br>
        Using {model_source} | TensorFlow {tf_version}
    </p>
</div>
""".format(model_source=model_source if model_source else "pre-trained model", tf_version=tf_version), 
unsafe_allow_html=True)

# Model format selector
st.sidebar.markdown("---")
st.sidebar.info("""
**Available Model Formats:**
1. ✅ `multitask_loan_model_tf/` (SavedModel - BEST)
2. ✅ `multitask_loan_model.h5` (H5 format)
3. ✅ `multitask_loan_model1.h5` (H5 format)
4. ✅ `multitask_loan_model.keras` (Keras format)
5. ✅ `multitask_loan_model1.keras` (Keras format)
""")

if st.sidebar.button("🔄 Refresh App"):
    st.rerun()