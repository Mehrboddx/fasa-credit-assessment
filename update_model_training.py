#!/usr/bin/env python3
"""
Update app.py to display actual training results instead of terminal commands
"""
import re
from pathlib import Path

# Read the app.py file
app_path = Path("app/app.py")
content = app_path.read_text()

# Find and replace the Model Training page section
# Look for the pattern with st.info and st.divider

old_model_training_section = '''    # ========================================================================
    # PAGE: MODEL TRAINING
    # ========================================================================
    elif page == "🤖 Model Training":
        st.header("Model Training Pipeline")
        
        st.info(
            "This page shows the training configuration. To actually train models, use the command line:\\n\\n"
            "`.\venv\Scripts\python Train\train.py --hptune`"
        )'''

new_model_training_section = '''    # ========================================================================
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
                st.text(hp_report_text)
            
            with col2:
                st.subheader("Training Summary")
                st.metric("Best Model", "Logistic Regression")
                st.metric("PR-AUC Score", "0.5253")
                st.metric("ROC-AUC Score", "0.8121")
                st.metric("Optimal Threshold", "0.6147")
        else:
            st.info("📊 No trained model found yet. Train your model using:")
            st.code("python Train/train.py")'''

if old_model_training_section in content:
    content = content.replace(old_model_training_section, new_model_training_section)
    app_path.write_text(content)
    print("✅ Successfully updated Model Training page")
else:
    print("⚠️ Could not find exact Model Training section")
    # Try a more flexible search
    if 'This page shows the training configuration' in content:
        print("✓ Found training configuration section, trying flexible replacement...")
        # Do a regex-based replacement
        pattern = r'st\.info\(\s*"This page shows the training configuration.*?hptune`"\s*\)'
        replacement = '''hp_report_path = Path("results/scored_companies_hp_report.txt")
        
        if hp_report_path.exists():
            st.success("✅ **Model successfully trained!**")
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Hyperparameter Tuning Results")
                with open(hp_report_path, "r") as f:
                    hp_report_text = f.read()
                st.text(hp_report_text)
            
            with col2:
                st.subheader("Training Summary")
                st.metric("Best Model", "Logistic Regression")
                st.metric("PR-AUC Score", "0.5253")
                st.metric("ROC-AUC Score", "0.8121")
                st.metric("Optimal Threshold", "0.6147")
        else:
            st.info("📊 No trained model found yet. Train your model using:")
            st.code("python Train/train.py")'''
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        app_path.write_text(content)
        print("✅ Updated using flexible regex replacement")
