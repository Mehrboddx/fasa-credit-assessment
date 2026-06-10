#!/usr/bin/env python
"""Fix the navigation list in app.py to include Predictions"""

with open('app/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the page selection radio button
found = False
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Look for the radio button definition
    if 'st.sidebar.radio' in line and '"Select Page"' in lines[i+1]:
        # Found it - replace the list on the next line
        new_lines.append(line)  # st.sidebar.radio line
        i += 1
        new_lines.append(lines[i])  # "Select Page" line
        i += 1
        
        # Now replace the list
        # Find the opening bracket
        while i < len(lines) and '[' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        
        # Replace the list with clean version
        new_lines.append('        [\n')
        new_lines.append('            "📊 Dashboard",\n')
        new_lines.append('            "🤖 Model Training",\n')
        new_lines.append('            "🔬 Experiment Comparison",\n')
        new_lines.append('            "📈 Results Explorer",\n')
        new_lines.append('            "📋 Predictions",\n')
        new_lines.append('            "💼 Credit Assessment",\n')
        new_lines.append('            "🏢 Company Analysis",\n')
        new_lines.append('            "❓ Help"\n')
        new_lines.append('        ]\n')
        
        # Skip old list lines
        while i < len(lines) and ']' not in lines[i]:
            i += 1
        i += 1  # skip the ]
        
        found = True
    else:
        new_lines.append(line)
        i += 1

if found:
    with open('app/app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("✓ Successfully added Predictions to navigation")
else:
    print("⚠ Could not find navigation section")

