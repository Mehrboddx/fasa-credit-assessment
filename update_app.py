import os

# Read the current app.py
with open("app/app.py", "r") as f:
    content = f.read()

# Find and replace the Model Training section that shows terminal command
old_model_training = '''    # ========================================================================
    # PAGE: MODEL TRAINING
    # ========================================================================
    elif page == "🤖 Model Training":
        st.header("Model Training Pipeline")
        
        st.info(
            "This page shows the training configuration. To actually train models, use the command line:\\n\\n"
            "`.\venv\Scripts\python Train\train.py --hptune`"
        )'''

new_model_training = '''    # ========================================================================
    # PAGE: MODEL TRAINING
    # ========================================================================
    elif page == "🤖 Model Training":
        st.header("Model Training Pipeline")
        
        # Load hyperparameter tuning report if available
        hp_report_path = Path("results/scored_companies_hp_report.txt")
        if hp_report_path.exists():
            st.success("✅ Model successfully trained! Below are the hyperparameter tuning results:")
            st.divider()
            
            with open(hp_report_path, "r") as f:
                hp_report_text = f.read()
            st.text(hp_report_text)
        else:
            st.info("📊 No trained model found yet. Run training to generate hyperparameter reports.")'''

if old_model_training in content:
    content = content.replace(old_model_training, new_model_training)
    with open("app/app.py", "w") as f:
        f.write(content)
    print("✅ Successfully updated Model Training section")
else:
    print("⚠️ Could not find exact Model Training section to replace")
    print("Searching for alternative patterns...")

