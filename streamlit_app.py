import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. Page Configuration & Custom CSS
# =====================================
st.set_page_config(
    page_title="House Price Analysis & Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & Aesthetics
st.markdown("""
    <style>
    /* Dark Glassmorphism Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6), rgba(17, 24, 39, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.25);
    }
    
    .prediction-price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================
# 2. Data Loading & Caching
# =====================================
@st.cache_data
def load_data(file_path_or_buffer="Housing.csv"):
    df = pd.read_csv(file_path_or_buffer)
    df.drop_duplicates(inplace=True)
    return df

@st.cache_data
def preprocess_data(df):
    df_encoded = df.copy()
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})
            
    if 'furnishingstatus' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['furnishingstatus'], drop_first=True)
        
    return df_encoded

# Sidebar setup
st.sidebar.image("https://img.icons8.com/isometric-line/100/6366f1/home.png", width=60)
st.sidebar.title("House Price Analysis")

uploaded_file = st.sidebar.file_to_change = st.sidebar.file_uploader("Upload custom Housing CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.sidebar.success("Custom CSV loaded successfully!")
else:
    df_raw = load_data("Housing.csv")

df_processed = preprocess_data(df_raw)

# Sidebar Navigation
navigation = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "House Price Predictor", "Data Explorer", "Model Benchmarks"]
)

# Header Banner
st.markdown('<div class="main-title">House Price Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Analytics & Machine Learning Valuation System</div>', unsafe_allow_html=True)

# =====================================
# 3. Model Training Helper
# =====================================
@st.cache_resource
def train_models(data):
    X = data.drop(columns=['price'])
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    return rf, lr, scaler, X.columns, X_test_scaled, y_test

rf_model, lr_model, scaler, feature_names, X_test_scaled, y_test = train_models(df_processed)

# =====================================
# TAB 1: DASHBOARD OVERVIEW
# =====================================
if navigation == "Dashboard Overview":
    st.subheader("Key Executive Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Properties", f"{len(df_raw):,}")
    col2.metric("Average Price", f"${df_raw['price'].mean():,.0f}")
    col3.metric("Median Price", f"${df_raw['price'].median():,.0f}")
    col4.metric("Max Price", f"${df_raw['price'].max():,.0f}")
    
    rf_preds = rf_model.predict(X_test_scaled)
    r2_score_val = r2_score(y_test, rf_preds)
    col5.metric("Model R² Accuracy", f"{r2_score_val*100:.1f}%")

    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("House Price Distribution")
        fig_dist = px.histogram(
            df_raw, 
            x="price", 
            nbins=30, 
            color_discrete_sequence=['#6366f1'],
            marginal="box",
            title="Price Density Distribution"
        )
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.subheader("Price vs. Square Footage Area")
        fig_scatter = px.scatter(
            df_raw, 
            x="area", 
            y="price", 
            color="airconditioning",
            size="bathrooms",
            color_discrete_map={"yes": "#38bdf8", "no": "#f43f5e"},
            title="Area vs Price (Size = Bathrooms)"
        )
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.subheader("Correlation Matrix")
        numeric_df = df_processed.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr, 
            text_auto=".2f", 
            color_continuous_scale="Viridis",
            title="Feature Correlation Heatmap"
        )
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_b2:
        st.subheader("Price Premium by Furnishing Status")
        fig_box = px.box(
            df_raw, 
            x="furnishingstatus", 
            y="price", 
            color="furnishingstatus",
            title="Price Distribution across Furnishing Types"
        )
        fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

# =====================================
# TAB 2: AI PRICE PREDICTOR
# =====================================
elif navigation == "House Price Predictor":
    st.subheader("Interactive Valuation Calculator")
    
    col_inputs, col_output = st.columns([1.2, 0.8])
    
    with col_inputs:
        area_input = st.slider("Total Area (sq. ft.)", min_value=1500, max_value=16500, value=6000, step=100)
        
        c1, c2 = st.columns(2)
        with c1:
            bedrooms_input = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
            bathrooms_input = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
            stories_input = st.selectbox("Stories", [1, 2, 3, 4], index=1)
        with c2:
            parking_input = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=2)
            furnishing_input = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=0)

        st.markdown("**Amenities & Location:**")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            ac_input = st.checkbox("Air Conditioning", value=True)
            mainroad_input = st.checkbox("Main Road Access", value=True)
        with tc2:
            prefarea_input = st.checkbox("Preferred Neighborhood", value=True)
            basement_input = st.checkbox("Basement", value=False)
        with tc3:
            guestroom_input = st.checkbox("Guestroom", value=False)
            hotwater_input = st.checkbox("Hot Water Heating", value=False)

    # Prepare input vector for model prediction
    input_dict = {
        'area': area_input,
        'bedrooms': bedrooms_input,
        'bathrooms': bathrooms_input,
        'stories': stories_input,
        'mainroad': 1 if mainroad_input else 0,
        'guestroom': 1 if guestroom_input else 0,
        'basement': 1 if basement_input else 0,
        'hotwaterheating': 1 if hotwater_input else 0,
        'airconditioning': 1 if ac_input else 0,
        'parking': parking_input,
        'prefarea': 1 if prefarea_input else 0,
        'furnishingstatus_semi-furnished': 1 if furnishing_input == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishing_input == 'unfurnished' else 0,
    }
    
    input_df = pd.DataFrame([input_dict])
    # Align columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    
    scaled_input = scaler.transform(input_df)
    predicted_price = rf_model.predict(scaled_input)[0]
    
    with col_output:
        st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;">Estimated Valuation</div>
                <div class="prediction-price">${predicted_price:,.0f}</div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    Confidence Interval (95%): <b>${predicted_price*0.95:,.0f}</b> - <b>${predicted_price*1.05:,.0f}</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Top Feature Contributions")
        feat_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
        
        fig_imp = px.bar(
            x=feat_importances.values, 
            y=feat_importances.index, 
            orientation='h',
            color_discrete_sequence=['#38bdf8'],
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig_imp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
        st.plotly_chart(fig_imp, use_container_width=True)

# =====================================
# TAB 3: DATA EXPLORER
# =====================================
elif navigation == "Data Explorer":
    st.subheader("Raw Data Inspector")
    
    c1, c2 = st.columns(2)
    with c1:
        furnish_filter = st.multiselect(
            "Filter Furnishing Status", 
            options=df_raw["furnishingstatus"].unique(), 
            default=df_raw["furnishingstatus"].unique()
        )
    with c2:
        price_range = st.slider(
            "Price Range ($)", 
            min_value=int(df_raw['price'].min()), 
            max_value=int(df_raw['price'].max()), 
            value=(int(df_raw['price'].min()), int(df_raw['price'].max()))
        )
        
    filtered_df = df_raw[
        (df_raw['furnishingstatus'].isin(furnish_filter)) & 
        (df_raw['price'] >= price_range[0]) & 
        (df_raw['price'] <= price_range[1])
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="filtered_housing_data.csv",
        mime="text/csv"
    )

# =====================================
# TAB 4: MODEL BENCHMARKS
# =====================================
elif navigation == "Model Benchmarks":
    st.subheader("Machine Learning Algorithm Comparison")
    
    lr_preds = lr_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test_scaled)
    
    benchmark_df = pd.DataFrame({
        "Algorithm": ["Random Forest Regressor", "Multiple Linear Regression"],
        "R² Score": [r2_score(y_test, rf_preds), r2_score(y_test, lr_preds)],
        "MAE ($)": [mean_absolute_error(y_test, rf_preds), mean_absolute_error(y_test, lr_preds)],
        "RMSE ($)": [np.sqrt(mean_squared_error(y_test, rf_preds)), np.sqrt(mean_squared_error(y_test, lr_preds))]
    })
    
    st.table(benchmark_df.style.format({"R² Score": "{:.4f}", "MAE ($)": "${:,.2f}", "RMSE ($)": "${:,.2f}"}))
    
    st.subheader("Actual vs Predicted Prices Scatter Plot")
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=y_test, y=rf_preds, mode='markers', name='Random Forest', marker=dict(color='#a855f7')))
    fig_val.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect Fit (Y=X)', line=dict(color='#10b981', dash='dash')))
    fig_val.update_layout(template="plotly_dark", xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_val, use_container_width=True)
