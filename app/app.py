"""
Fasanara Credit Assessment App
Interactive Streamlit interface for credit risk modeling and scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent directory to path so we can import train module
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="Fasanara Credit Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        color: #e74c3c;
        font-weight: bold;
    }
    .low-risk {
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_data_files(data_path: str = "data"):
    """Load training and scoring data"""
    data_dir = Path(data_path)
    
    try:
        train_companies = pd.read_csv(data_dir / "train_companies.csv", index_col=0)
        train_narratives = pd.read_csv(data_dir / "train_narratives.csv", index_col=0)
        train_outcomes = pd.read_csv(data_dir / "train_outcomes.csv", index_col=0)
        scoring_companies = pd.read_csv(data_dir / "scoring_companies.csv", index_col=0)
        
        # Reset indices to make company_id a proper column
        train_companies = train_companies.reset_index()
        train_narratives = train_narratives.reset_index()
        train_outcomes = train_outcomes.reset_index()
        scoring_companies = scoring_companies.reset_index()
        
        # Merge training data
        train_data = train_companies.merge(
            train_outcomes[["company_id", "defaulted"]], 
            on="company_id", 
            how="left"
        )
        train_data = train_data.merge(
            train_narratives[["company_id", "business_description"]],
            on="company_id",
            how="left"
        )
        
        return train_data, scoring_companies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def plot_sector_analysis(train_data):
    """Create sector analysis visualization"""
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Count by sector
    ax = axes[0]
    sector_counts = train_data["sector"].value_counts()
    sector_counts.plot(kind='barh', ax=ax, color='#3498db', edgecolor='black', linewidth=1)
    ax.set_xlabel("Number of Companies", fontsize=11, fontweight='bold')
    ax.set_ylabel("Sector", fontsize=11, fontweight='bold')
    ax.set_title("Company Count by Sector", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Default rate by sector
    ax = axes[1]
    sector_default_rates = train_data.groupby("sector")["defaulted"].mean().sort_values(ascending=False)
    colors = ['#e74c3c' if x > train_data["defaulted"].mean() else '#2ecc71' 
              for x in sector_default_rates.values]
    sector_default_rates.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1)
    ax.axvline(train_data["defaulted"].mean(), color='black', linestyle='--', linewidth=2, label='Overall Default Rate')
    ax.set_xlabel("Default Rate", fontsize=11, fontweight='bold')
    ax.set_ylabel("Sector", fontsize=11, fontweight='bold')
    ax.set_title("Default Rate by Sector", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_country_analysis(train_data):
    """Create country analysis visualization"""
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Count by country
    ax = axes[0]
    country_counts = train_data["country"].value_counts()
    country_counts.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='black', linewidth=1)
    ax.set_xlabel("Number of Companies", fontsize=11, fontweight='bold')
    ax.set_ylabel("Country", fontsize=11, fontweight='bold')
    ax.set_title("Company Count by Country", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Default rate by country
    ax = axes[1]
    country_default_rates = train_data.groupby("country")["defaulted"].mean().sort_values(ascending=False)
    colors = ['#e74c3c' if x > train_data["defaulted"].mean() else '#2ecc71' 
              for x in country_default_rates.values]
    country_default_rates.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1)
    ax.axvline(train_data["defaulted"].mean(), color='black', linestyle='--', linewidth=2, label='Overall Default Rate')
    ax.set_xlabel("Default Rate", fontsize=11, fontweight='bold')
    ax.set_ylabel("Country", fontsize=11, fontweight='bold')
    ax.set_title("Default Rate by Country", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_feature_correlations(train_data):
    """Create feature correlation visualization"""
    if not HAS_MATPLOTLIB:
        return None
    
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["company_id", "defaulted"]]
    
    correlations = train_data[numeric_cols + ["defaulted"]].corr()["defaulted"].drop("defaulted")
    correlations = correlations.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in correlations.values]
    correlations.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel("Correlation with Default", fontsize=11, fontweight='bold')
    ax.set_title("Feature Correlations with Default Target", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


@st.cache_resource
def load_feature_importance():
    """Load feature importance from trained model"""
    import pickle
    try:
        importance_path = Path("results/feature_importance.pkl")
        if importance_path.exists():
            with open(importance_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    return None


@st.cache_resource
def load_hp_report():
    """Load hyperparameter tuning report"""
    try:
        report_path = Path("results/scored_companies_hp_report.txt")
        if report_path.exists():
            with open(report_path, "r") as f:
                return f.read()
    except Exception as e:
        st.warning(f"Could not load HP report: {e}")
    return None


@st.cache_resource
def load_consolidated_results():
    """Load consolidated experiment comparison results"""
    try:
        csv_path = Path("experiments/CONSOLIDATED_RESULTS.csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Could not load consolidated results: {e}")
    return None


@st.cache_resource
def load_narrative_embeddings_2d():
    """Load 2D PCA narrative embeddings"""
    try:
        embeddings_path = Path("outputs/Analytics/narrative_embeddings_2d_pca_cleaned.csv")
        if embeddings_path.exists():
            return pd.read_csv(embeddings_path)
    except Exception as e:
        st.warning(f"Could not load narrative embeddings: {e}")
    return None


@st.cache_resource
def load_training_data_for_embeddings():
    """Load training data to match with embeddings for coloring"""
    try:
        train_companies = pd.read_csv("data/train_companies.csv")
        train_outcomes = pd.read_csv("data/train_outcomes.csv")
        return train_companies.merge(train_outcomes, on="company_id", how="left")
    except Exception as e:
        st.warning(f"Could not load training data: {e}")
    return None


def plot_narrative_embeddings_interactive(embeddings_df, train_data):
    """Create interactive narrative embeddings visualization"""
    if embeddings_df is None or train_data is None or not HAS_MATPLOTLIB:
        return None
    
    # Merge embeddings with default status
    merged = embeddings_df.merge(train_data[["company_id", "defaulted"]], on="company_id", how="left")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot by default status
    colors = {0: '#2ecc71', 1: '#e74c3c'}  # Green for safe, red for default
    labels = {0: 'Safe', 1: 'Defaulted'}
    
    for default_status in [0, 1]:
        mask = merged['defaulted'] == default_status
        data = merged[mask]
        ax.scatter(
            data['embedding_0'],
            data['embedding_1'],
            c=colors[default_status],
            label=labels[default_status],
            alpha=0.6,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('First Principal Component (PCA)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Second Principal Component (PCA)', fontsize=12, fontweight='bold')
    ax.set_title('Narrative Embeddings Visualization (2D PCA)\nBusiness Description Semantic Space', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_importance):
    """Create feature importance visualization"""
    if not HAS_MATPLOTLIB or feature_importance is None:
        return None
    
    # Sort by importance descending
    imp_sorted = feature_importance.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(imp_sorted)), imp_sorted.values, color='#3498db', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted.index)
    ax.set_xlabel("Importance Score", fontsize=11, fontweight='bold')
    ax.set_title("Top 10 Features - Model Importance", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("# 💰 Fasanara Credit Assessment System")
    st.markdown("*AI-powered credit risk modeling with narrative analysis*")
    st.divider()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "📊 Dashboard",
            "🤖 Model Training",
            "🔬 Experiment Comparison",
            "📈 Results Explorer",
            "📋 Predictions",
            "💼 Credit Assessment",
            "🏢 Company Analysis",
            "❓ Help"
        ]
    )
    
    # Load data once
    train_data, scoring_companies = load_data_files()
    
    if train_data is None or scoring_companies is None:
        st.error("Failed to load data. Please check the data directory.")
        return
    
    # ========================================================================
    # PAGE: DASHBOARD
    # ========================================================================
    if page == "📊 Dashboard":
        st.header("Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Companies", len(train_data))
        with col2:
            st.metric("Defaulted", train_data["defaulted"].sum())
        with col3:
            st.metric("Default Rate", f"{train_data['defaulted'].mean():.1%}")
        with col4:
            st.metric("Sectors", train_data["sector"].nunique())
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sector Analysis", "Country Analysis", "Feature Correlations", "Model Features", "Narrative Embeddings"])
        
        with tab1:
            st.subheader("Sector Risk Profile")
            sector_analysis = train_data.groupby("sector").agg({
                "company_id": "count",
                "defaulted": ["sum", "mean"]
            }).round(4)
            sector_analysis.columns = ["count", "defaults", "default_rate"]
            sector_analysis = sector_analysis.sort_values("count", ascending=False)
            
            st.dataframe(sector_analysis, use_container_width=True)
            
            # Load pre-generated sector analysis chart
            sector_png = Path("outputs/Analytics/sector_analysis.png")
            if sector_png.exists():
                from PIL import Image
                img = Image.open(sector_png)
                st.image(img, use_container_width=True)
            else:
                st.info("📊 Sector analysis visualization will be generated during model training.")
        
        with tab2:
            st.subheader("Country Risk Profile")
            country_analysis = train_data.groupby("country").agg({
                "company_id": "count",
                "defaulted": ["sum", "mean"]
            }).round(4)
            country_analysis.columns = ["count", "defaults", "default_rate"]
            country_analysis = country_analysis.sort_values("default_rate", ascending=False)
            
            st.dataframe(country_analysis, use_container_width=True)
            
            # Load pre-generated country analysis chart
            country_png = Path("outputs/Analytics/country_analysis.png")
            if country_png.exists():
                from PIL import Image
                img = Image.open(country_png)
                st.image(img, use_container_width=True)
            else:
                st.info("📊 Country analysis visualization will be generated during model training.")
        
        with tab3:
            st.subheader("Feature Importance")
            
            # Load pre-generated correlation heatmap
            corr_png = Path("outputs/Analytics/correlation_heatmap.png")
            if corr_png.exists():
                from PIL import Image
                img = Image.open(corr_png)
                st.image(img, use_container_width=True)
            else:
                st.info("📊 Correlation heatmap will be generated during model training.")
        
        with tab4:
            st.subheader("Top 10 Model Features")
            feature_importance = load_feature_importance()
            if feature_importance is not None:
                st.write("**Feature importance ranking** from the trained credit risk model")
                
                # Display as dataframe
                imp_df = pd.DataFrame({
                    "Feature": feature_importance.index,
                    "Importance Score": feature_importance.values
                }).reset_index(drop=True)
                imp_df.index = imp_df.index + 1
                st.dataframe(imp_df, use_container_width=True)
            else:
                st.info("📊 No trained model found. Run `python Train/train.py` to train the model and generate feature importances.")
        
        with tab5:
            st.subheader("Narrative Embeddings Visualization")
            st.markdown("""
            This visualization shows how AI embeddings represent company business descriptions 
            in a 2D semantic space using Principal Component Analysis (PCA).
            
            - **Green points** = Safe companies (no default)
            - **Red points** = Defaulted companies
            
            Companies with similar descriptions appear near each other. The spatial separation 
            between red and green points indicates how well narrative embeddings can distinguish 
            between safe and defaulted companies.
            """)
            
            col_emb1, col_emb2 = st.columns(2)
            
            # Load non-cleaned embedding visualization
            embedding_png = Path("outputs/Analytics/narrative_embeddings_2d.png")
            if embedding_png.exists():
                from PIL import Image
                img = Image.open(embedding_png)
                with col_emb1:
                    st.image(img, caption="All Data Points", use_container_width=True)
            else:
                with col_emb1:
                    st.info("📊 Visualization will be generated during model training.")
            
            # Load cleaned embedding visualization
            embedding_cleaned_png = Path("outputs/Analytics/narrative_embeddings_2d_cleaned.png")
            if embedding_cleaned_png.exists():
                from PIL import Image
                img_cleaned = Image.open(embedding_cleaned_png)
                with col_emb2:
                    st.image(img_cleaned, caption="Cleaned (Outliers Removed)", use_container_width=True)
            else:
                with col_emb2:
                    st.info("📊 Visualization will be generated during model training.")
            
            st.divider()
            st.markdown("""
            **Embedding Statistics:**
            - Model: all-MiniLM-L6-v2 (384-dimensional embeddings)
            - Dimensionality Reduction: PCA (Principal Component Analysis)
            - Dimensions shown: First 2 principal components
            """)
    
    # ========================================================================
    # PAGE: COMPANY ANALYSIS
    # ========================================================================
    elif page == "🏢 Company Analysis":
        st.header("Company Analysis")
        
        st.subheader("Select a Company")
        selected_company = st.selectbox(
            "Choose a company from training set:",
            options=train_data["company_name"].unique(),
            key="company_select"
        )
        
        company = train_data[train_data["company_name"] == selected_company].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Info")
            st.info(f"""
            **ID:** {company['company_id']}
            **Sector:** {company['sector']}
            **Country:** {company['country']}
            **Years in Operation:** {int(company['years_in_operation'])}
            **Default Status:** {'🔴 DEFAULTED' if company['defaulted'] == 1 else '🟢 NON-DEFAULT'}
            """)
        
        with col2:
            st.subheader("Financial Metrics")
            st.metric("Revenue (€M)", f"{company['revenue_m']:.1f}")
            st.metric("EBITDA Margin", f"{company['ebitda_margin']:.1%}")
            st.metric("Debt Ratio", f"{company['debt_ratio']:.2f}")
            st.metric("Interest Coverage", f"{company['interest_coverage']:.2f}x")
            st.metric("Cash Ratio", f"{company['cash_ratio']:.3f}")
        
        st.divider()
        st.subheader("Business Description")
        st.write(company["business_description"])
    
    # ========================================================================
    # PAGE: PREDICTIONS
    # ========================================================================
    elif page == "📋 Predictions":
        st.header("Prediction Results")
        st.markdown("*Credit risk predictions and assessment details*")
        
        predictions_path = Path("outputs/predictions.csv")
        
        if predictions_path.exists():
            # Load predictions
            predictions_df = pd.read_csv(predictions_path)
            
            st.divider()
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(predictions_df))
            
            with col2:
                avg_risk = predictions_df['predicted_default_probability'].mean()
                st.metric("Average Risk Probability", f"{avg_risk:.1%}")
            
            with col3:
                high_risk = (predictions_df['predicted_default_probability'] > 0.5).sum()
                st.metric("High Risk Companies", high_risk)
            
            with col4:
                low_risk = (predictions_df['predicted_default_probability'] <= 0.5).sum()
                st.metric("Low Risk Companies", low_risk)
            
            st.divider()
            
            # Filters
            st.subheader("🔍 Filter & Search")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_prob = st.slider("Min Risk Probability", 0.0, 1.0, 0.0)
            
            with col2:
                max_prob = st.slider("Max Risk Probability", 0.0, 1.0, 1.0)
            
            with col3:
                risk_filter = st.selectbox(
                    "Filter by Risk Rating",
                    ["All"] + sorted(predictions_df['risk_rating'].unique().tolist())
                )
            
            # Apply filters
            filtered_df = predictions_df[
                (predictions_df['predicted_default_probability'] >= min_prob) &
                (predictions_df['predicted_default_probability'] <= max_prob)
            ].copy()
            
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df['risk_rating'] == risk_filter]
            
            # Sort by risk probability descending
            filtered_df = filtered_df.sort_values('predicted_default_probability', ascending=False)
            
            st.divider()
            
            st.subheader(f"Results ({len(filtered_df)} companies)")
            
            # Display predictions table
            display_df = filtered_df[['company_id', 'predicted_default_probability', 'risk_rating']].copy()
            display_df.columns = ['Company ID', 'Default Probability', 'Risk Rating']
            display_df['Default Probability'] = display_df['Default Probability'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Display detailed explanations
            st.subheader("📋 Detailed Assessments")
            
            selected_company = st.selectbox(
                "Select a company to view detailed credit memo:",
                options=filtered_df['company_id'].values
            )
            
            if selected_company:
                company_data = filtered_df[filtered_df['company_id'] == selected_company].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Company ID", company_data['company_id'])
                
                with col2:
                    risk_prob = company_data['predicted_default_probability']
                    st.metric("Default Probability", f"{risk_prob:.1%}")
                
                with col3:
                    st.metric("Risk Rating", company_data['risk_rating'])
                
                st.divider()
                
                st.subheader("💬 Credit Assessment Memo")
                st.markdown(company_data['explanation'])
            
            st.divider()
            
            # Download predictions
            st.subheader("📥 Export Data")
            
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Results (CSV)",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon="📊"
            )
        
        else:
            st.info("""
            📊 **No predictions found yet.**
            
            To generate predictions, run:
            ```bash
            python Train/train.py --memos
            ```
            
            This will:
            - Train the credit risk model
            - Generate predictions for all scoring companies
            - Create detailed credit assessment memos
            - Save results to `outputs/predictions.csv`
            """)
    
    # ========================================================================
    # PAGE: CREDIT ASSESSMENT
    # ========================================================================
    elif page == "💼 Credit Assessment":
        st.header("Company Credit Assessment")
        st.markdown("*Submit your company information for AI-powered credit risk evaluation*")
        
        # Check if model exists
        model_path = Path("results/best_model.pkl")
        if not model_path.exists():
            st.warning("⚠️ No trained model found. Please train the model first using the 'Model Training' page.")
            st.info("Run: `python Train/train.py` in the terminal to generate the model.")
            st.stop()
        
        st.divider()
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Financial Metrics")
            company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
            revenue_m = st.number_input("Revenue (€M)", min_value=0.1, step=1.0, value=10.0)
            ebitda_margin = st.number_input("EBITDA Margin (%)", min_value=-100.0, max_value=100.0, step=1.0, value=15.0)
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=10.0, step=0.1, value=0.5)
            interest_coverage = st.number_input("Interest Coverage (x)", min_value=0.1, step=0.1, value=3.0)
            cash_ratio = st.number_input("Cash Ratio", min_value=0.0, max_value=5.0, step=0.1, value=0.3)
        
        with col2:
            st.subheader("🏭 Company Information")
            years_op = st.number_input("Years in Operation", min_value=0, max_value=100, step=1, value=5)
            employee_count = st.number_input("Employee Count", min_value=1, max_value=100000, step=10, value=50)
            revenue_growth = st.number_input("Revenue Growth YoY (%)", min_value=-100.0, max_value=100.0, step=1.0, value=5.0)
            country = st.selectbox("Country", options=sorted(train_data["country"].unique()), index=0)
            sector = st.selectbox("Sector", options=sorted(train_data["sector"].unique()), index=0)
        
        st.divider()
        st.subheader("📝 Company Description")
        description_input = st.text_area(
            "Business Description & Risk Factors",
            placeholder="Describe your company: business model, market position, risk factors, key strengths, etc.",
            height=200
        )
        
        st.divider()
        
        # Assessment button
        if st.button("🔍 Run Credit Assessment", use_container_width=True, type="primary"):
            if not company_name:
                st.error("Please enter a company name.")
                st.stop()
            
            if not description_input or len(description_input.strip()) < 20:
                st.error("Please enter a detailed company description (minimum 20 characters).")
                st.stop()
            
            with st.spinner("🔄 Analyzing company..."):
                try:
                    # Import necessary functions from train.py
                    from Train.train import (
                        engineer_quant_features, 
                        fit_target_encoder, 
                        apply_target_encoder,
                        get_feature_columns,
                        get_risk_rating,
                        generate_credit_memo,
                    )
                    import pickle
                    
                    # Create company dataframe
                    company_data = pd.DataFrame({
                        "company_id": [9999],
                        "company_name": [company_name],
                        "revenue_m": [revenue_m],
                        "ebitda_margin": [ebitda_margin / 100],  # Convert percentage to decimal
                        "debt_ratio": [debt_ratio],
                        "interest_coverage": [interest_coverage],
                        "cash_ratio": [cash_ratio],
                        "years_in_operation": [years_op],
                        "employee_count": [employee_count],
                        "revenue_growth": [revenue_growth / 100],  # Convert percentage to decimal
                        "country": [country],
                        "sector": [sector],
                        "business_description": [description_input],
                        "n_risks": [description_input.count("risk") + description_input.count("issue") + description_input.count("challenge")],
                        "n_benefits": [description_input.count("strength") + description_input.count("opportunity") + description_input.count("advantage")],
                    })
                    
                    # Engineer features
                    company_data = engineer_quant_features(company_data)
                    
                    # Load encoders and apply target encoding
                    encoders_path = Path("results/target_encoders.pkl")
                    if encoders_path.exists():
                        with open(encoders_path, "rb") as f:
                            encoders = pickle.load(f)
                        company_data = apply_target_encoder(company_data, encoders, ["country", "sector"])
                    else:
                        # Use fallback: mean encoding from training data
                        encoders = fit_target_encoder(train_data, ["country", "sector"])
                        company_data = apply_target_encoder(company_data, encoders, ["country", "sector"])
                    
                    # Load trained model
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    
                    # Load feature columns from training
                    cols_path = Path("results/feature_columns.pkl")
                    if cols_path.exists():
                        with open(cols_path, "rb") as f:
                            feature_cols = pickle.load(f)
                    else:
                        # Fallback: compute from company_data
                        feature_cols = get_feature_columns(company_data)
                    
                    X = company_data[feature_cols].fillna(0)
                    
                    # Get prediction
                    prob = model.predict_proba(X)[0, 1]
                    pred = int(model.predict(X)[0])
                    
                    # Display results
                    st.divider()
                    st.subheader("📋 Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_rating = get_risk_rating(prob)
                        st.metric("Risk Rating", risk_rating)
                    
                    with col2:
                        st.metric("Default Probability", f"{prob:.1%}")
                    
                    with col3:
                        status = "⚠️ REJECTED" if pred == 1 else "✅ APPROVED"
                        st.metric("Credit Decision", status)
                    
                    st.divider()
                    
                    # Risk assessment details
                    st.subheader("🎯 Risk Assessment")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Debt Ratio**: {debt_ratio:.2f}")
                        st.write(f"**EBITDA Margin**: {ebitda_margin:.1f}%")
                        st.write(f"**Interest Coverage**: {interest_coverage:.2f}x")
                    
                    with col2:
                        st.write(f"**Cash Ratio**: {cash_ratio:.3f}")
                        st.write(f"**Revenue (€M)**: €{revenue_m:.1f}M")
                        st.write(f"**Employees**: {employee_count:,}")
                    
                    st.divider()
                    
                    # Generate credit memo if API key available
                    st.subheader("💬 Credit Analysis Memo")
                    
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if api_key:
                        with st.spinner("📝 Generating analysis memo..."):
                            try:
                                import anthropic
                                client = anthropic.Anthropic(api_key=api_key)
                                memo = generate_credit_memo(client, company_data.iloc[0], prob, pred)
                                st.markdown(memo)
                                
                                # Download memo
                                st.download_button(
                                    label="📥 Download Memo",
                                    data=memo,
                                    file_name=f"credit_memo_{company_name.replace(' ', '_')}.txt",
                                    mime="text/plain"
                                )
                            except Exception as e:
                                st.error(f"Failed to generate memo: {e}")
                    else:
                        st.info("💡 Set `ANTHROPIC_API_KEY` environment variable to generate detailed analysis memos.")
                        
                        # Fallback simple explanation
                        if pred == 1:
                            st.warning(f"**Higher Risk Profile** - Probability of default: {prob:.1%}")
                        else:
                            st.success(f"**Favorable Risk Profile** - Probability of default: {prob:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Assessment failed: {str(e)}")
                    st.exception(e)
    
    # ========================================================================
    # PAGE: MODEL TRAINING
    # ========================================================================
    elif page == "🤖 Model Training":
        st.header("Model Training Pipeline")
        
        # Load hyperparameter report
        hp_report_path = Path("results/scored_companies_hp_report.txt")
        
        if hp_report_path.exists():
            st.success("✅ **Model successfully trained!**")
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Hyperparameter Tuning Results")
                with open(hp_report_path, "r") as f:
                    hp_report_text = f.read()
                
                # Parse and display HP report beautifully
                lines = hp_report_text.strip().split("\n")
                models_hp = {}
                current_model = None
                
                for line in lines:
                    if line.startswith("[") and line.endswith("]"):
                        current_model = line.strip("[]")
                        models_hp[current_model] = {}
                    elif "=" in line and current_model:
                        param, value = line.split("=")
                        param = param.strip()
                        value = value.strip()
                        models_hp[current_model][param] = value
                
                # Display in tabs
                if models_hp:
                    hp_tabs = st.tabs(list(models_hp.keys()))
                    for tab, (model_name, params) in zip(hp_tabs, models_hp.items()):
                        with tab:
                            cols = st.columns(2)
                            for idx, (param, value) in enumerate(params.items()):
                                with cols[idx % 2]:
                                    st.metric(param, value)
                else:
                    st.text(hp_report_text)
            
            with col2:
                st.subheader("Training Summary")
                st.metric("Best Model", "Logistic Regression")
                st.metric("PR-AUC Score", "0.5253")
                st.metric("ROC-AUC Score", "0.8121")
                st.metric("Optimal Threshold", "0.6147")
        else:
            st.info("📊 No trained model found yet. Train your model using:")
            st.code("python Train/train.py")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"]
            )
            finetune = st.checkbox("Enable Fine-tuning", value=False)
            if finetune:
                finetune_frac = st.slider("Fine-tune Fraction", 0.1, 0.5, 0.4)
                finetune_epochs = st.slider("Fine-tune Epochs", 1, 10, 3)
        
        with col2:
            st.subheader("Hyperparameter Search")
            hptune = st.checkbox("Enable Hyperparameter Tuning", value=False)
            if hptune:
                hptune_trials = st.slider("Optuna Trials per Model", 10, 200, 50)
                hptune_inner_folds = st.slider("Inner CV Folds", 2, 5, 3)
        
        st.divider()
        
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", len(train_data))
        with col2:
            st.metric("Defaulted", train_data["defaulted"].sum())
        with col3:
            st.metric("Safe", (train_data["defaulted"] == 0).sum())
        with col4:
            st.metric("Default Rate", f"{train_data['defaulted'].mean():.1%}")
        
        st.warning("""
        **To start training:** Use the terminal command above with your preferred flags.
        Training takes 10-30 minutes depending on data size and hyperparameter search settings.
        """)
    
    # ========================================================================
    # PAGE: EXPERIMENT COMPARISON
    # ========================================================================
    elif page == "🔬 Experiment Comparison":
        st.header("Experiment Comparison Dashboard")
        st.markdown("""
        Compare performance across different experimental configurations:
        - **Baselines**: Statistical features only vs with embeddings
        - **Transformers**: Different sentence transformer models
        - **Fine-tuning**: Impact of embedding fine-tuning
        - **HP Tuning**: Effect of hyperparameter optimization
        """)
        
        st.divider()
        
        # Load consolidated results
        consolidated_df = load_consolidated_results()
        
        if consolidated_df is None or consolidated_df.empty:
            st.info("""
            📊 **No experiment results found yet.**
            
            To generate comprehensive experiment results, run:
            ```bash
            python run_full_experiment.py
            ```
            
            This will run 9 experiments comparing:
            1. Baseline (no embeddings)
            2-4. Three transformers (pretrained, no tuning)
            5-7. Three transformers (pretrained + HP tuning)
            8. Best transformer (fine-tuned)
            9. Best transformer (fine-tuned + HP tuned)
            """)
        else:
            # Display summary metrics
            st.subheader("📊 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_roc = consolidated_df['roc_auc'].max()
                st.metric("Best ROC-AUC", f"{best_roc:.4f}")
            
            with col2:
                avg_roc = consolidated_df['roc_auc'].mean()
                st.metric("Average ROC-AUC", f"{avg_roc:.4f}")
            
            with col3:
                best_pr = consolidated_df['pr_auc'].max()
                st.metric("Best PR-AUC", f"{best_pr:.4f}")
            
            with col4:
                total_exp = len(consolidated_df)
                st.metric("Total Experiments", total_exp)
            
            st.divider()
            
            # Grouped display by category
            st.subheader("📈 Results by Category")
            
            categories = consolidated_df['category'].unique()
            
            for category in categories:
                with st.expander(f"🔹 {category}", expanded=(category == "Best Transformer (Fine-tuned + HP Tuning)")):
                    cat_df = consolidated_df[consolidated_df['category'] == category].copy()
                    cat_df = cat_df.sort_values('roc_auc', ascending=False)
                    
                    # Format display columns
                    display_df = cat_df[[
                        'description',
                        'best_model',
                        'roc_auc',
                        'roc_auc_std',
                        'pr_auc',
                        'pr_auc_std',
                        'threshold',
                        'finetune',
                        'hptune'
                    ]].copy()
                    
                    display_df.columns = [
                        'Experiment',
                        'Best Model',
                        'ROC-AUC',
                        '±Std',
                        'PR-AUC',
                        '±Std',
                        'Threshold',
                        'Fine-tuned',
                        'HP-Tuned'
                    ]
                    
                    # Format numeric columns
                    display_df['ROC-AUC'] = display_df['ROC-AUC'].apply(lambda x: f"{x:.4f}")
                    display_df['±Std'] = display_df['±Std'].apply(lambda x: f"{x:.4f}")
                    display_df['PR-AUC'] = display_df['PR-AUC'].apply(lambda x: f"{x:.4f}")
                    display_df['±Std'] = display_df['±Std'].apply(lambda x: f"{x:.4f}")
                    display_df['Threshold'] = display_df['Threshold'].apply(lambda x: f"{x:.4f}")
                    display_df['Fine-tuned'] = display_df['Fine-tuned'].apply(lambda x: "✅" if x else "❌")
                    display_df['HP-Tuned'] = display_df['HP-Tuned'].apply(lambda x: "✅" if x else "❌")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Comparative charts
            st.subheader("📊 Performance Comparison")
            
            if HAS_MATPLOTLIB:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Sort by ROC-AUC for display
                plot_df = consolidated_df.sort_values('roc_auc', ascending=True)
                
                # ROC-AUC comparison
                ax = axes[0]
                x_pos = range(len(plot_df))
                ax.barh(x_pos, plot_df['roc_auc'], xerr=plot_df['roc_auc_std'], 
                       color='#3498db', alpha=0.7, edgecolor='black', linewidth=1, capsize=5)
                ax.set_yticks(x_pos)
                ax.set_yticklabels([d.split(' - ')[-1][:30] for d in plot_df['description']], fontsize=9)
                ax.set_xlabel('ROC-AUC', fontsize=11, fontweight='bold')
                ax.set_title('ROC-AUC Comparison (with std dev)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # PR-AUC comparison
                ax = axes[1]
                ax.barh(x_pos, plot_df['pr_auc'], xerr=plot_df['pr_auc_std'],
                       color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1, capsize=5)
                ax.set_yticks(x_pos)
                ax.set_yticklabels([d.split(' - ')[-1][:30] for d in plot_df['description']], fontsize=9)
                ax.set_xlabel('PR-AUC', fontsize=11, fontweight='bold')
                ax.set_title('PR-AUC Comparison (with std dev)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.divider()
            
            # Download consolidated results
            st.subheader("📥 Export Results")
            
            # CSV download
            csv = consolidated_df.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name=f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                icon="📊"
            )
            
            # Text download
            text_path = Path("experiments/CONSOLIDATED_RESULTS.txt")
            if text_path.exists():
                with open(text_path, 'r') as f:
                    text_content = f.read()
                st.download_button(
                    label="Download Report (TXT)",
                    data=text_content,
                    file_name=f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    icon="📝"
                )
    
    # ========================================================================
    # PAGE: RESULTS EXPLORER
    # ========================================================================
    elif page == "📈 Results Explorer":
        st.header("Scoring Results Explorer")
        
        # Check if results exist
        results_path = Path("results/scored_companies.csv")
        
        if results_path.exists():
            results = pd.read_csv(results_path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Companies Scored", len(results))
            with col2:
                st.metric("Predicted Default", results["predicted_default"].sum())
            with col3:
                st.metric("Predicted Safe", (results["predicted_default"] == 0).sum())
            with col4:
                st.metric("Avg Default Prob", f"{results['default_probability'].mean():.1%}")
            
            st.divider()
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_prob = st.slider("Min Default Probability", 0.0, 1.0, 0.0)
            with col2:
                max_prob = st.slider("Max Default Probability", 0.0, 1.0, 1.0)
            with col3:
                show_predicted_default = st.checkbox("Show Only Predicted Defaults", False)
            
            # Filter results
            filtered = results[
                (results["default_probability"] >= min_prob) &
                (results["default_probability"] <= max_prob)
            ]
            if show_predicted_default:
                filtered = filtered[filtered["predicted_default"] == 1]
            
            filtered = filtered.sort_values("default_probability", ascending=False)
            
            st.subheader(f"Results ({len(filtered)} companies)")
            st.dataframe(
                filtered[["company_name", "sector", "country", "default_probability", "predicted_default"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Download results
            st.divider()
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="Download Filtered Results (CSV)",
                data=csv,
                file_name=f"scored_companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Check for memos
            memos_path = Path("results/scored_companies_memos.txt")
            if memos_path.exists():
                st.divider()
                st.subheader("Credit Memos")
                
                selected_company_name = st.selectbox(
                    "View memo for company:",
                    options=filtered["company_name"].unique()
                )
                
                with open(memos_path, 'r') as f:
                    content = f.read()
                
                # Parse memos (simple approach)
                if selected_company_name in content:
                    st.info(f"Memo available for {selected_company_name}")
                    st.text("Open memos.txt file to view detailed credit analysis")
        else:
            st.warning("""
            No scoring results found. 
            Please run the training pipeline first:
            ```bash
            .\\venv\\Scripts\\python Train\\train.py
            ```
            """)
    
    # ========================================================================
    # PAGE: HELP
    # ========================================================================
    elif page == "❓ Help":
        st.header("Help & Documentation")
        
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Analyze Data** → Go to Dashboard to explore sector/country risk profiles
        2. **Train Model** → Use Model Training page for configuration, then run in terminal
        3. **Score Companies** → Pipeline produces predictions automatically
        4. **Review Results** → Explore predictions in Results Explorer
        """)
        
        st.divider()
        st.subheader("📊 Available Features")
        st.markdown("""
        - **Structured Features**: Revenue, debt ratio, EBITDA margin, interest coverage, cash ratio, etc.
        - **Narrative Embeddings**: 384-dimensional sentence embeddings from business descriptions
        - **Risk Signals**: Parsed risk/benefit factors from narratives
        - **Target Encoding**: Country and sector default rates (smoothed)
        - **Engineered Features**: Debt service risk, liquidity ratios, distress score
        """)
        
        st.divider()
        st.subheader("🤖 Model Options")
        st.markdown("""
        - **Logistic Regression**: Fast, interpretable baseline
        - **Random Forest**: Ensemble with feature importance
        - **XGBoost**: State-of-the-art gradient boosting
        - **Nested CV**: Separate hyperparameter tuning and model selection
        - **Threshold Optimization**: F1-based threshold tuning
        """)
        
        st.divider()
        st.subheader("⚙️ Terminal Commands")
        st.code("""
# Standard training (no hyperparameter search)
python Train/train.py

# With hyperparameter optimization
python Train/train.py --hptune

# With embedding fine-tuning
python Train/train.py --finetune

# With both fine-tuning and HP search
python Train/train.py --finetune --hptune

# Generate credit memos (requires ANTHROPIC_API_KEY)
python Train/train.py --memos

# All options combined
python Train/train.py --finetune --hptune --memos \\
  --finetune_epochs 5 --hptune_trials 100
        """)
        
        st.divider()
        st.subheader("📁 Output Files")
        st.markdown("""
        **Results/**
        - `scored_companies.csv` - Predictions and probabilities
        - `scored_companies.json` - Same in JSON format
        - `scored_companies_memos.txt` - Credit analysis memos
        - `scored_companies_hp_report.txt` - Hyperparameter tuning details
        - `scored_companies_embedding_comparison.txt` - Fine-tuning comparison
        
        **Outputs/**
        - `feature_correlations.csv` - Correlation analysis
        - `sector_analysis.csv` - Sector risk profiles
        - `country_analysis.csv` - Country risk profiles
        - `narrative_embeddings_2d.png` - PCA visualization
        - `narrative_embeddings_full_384d.csv` - Raw embeddings
        """)


if __name__ == "__main__":
    main()
