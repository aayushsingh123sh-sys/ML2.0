import streamlit as st
import pandas as pd
import numpy as np
import math

# --- STEP 1: ADD STREAMLIT TITLE ---
st.title("🌳 ID3 Decision Tree Practical")
st.write("This app runs your ID3 algorithm and displays the results on the web.")

# -----------------------------
# Dataset
# -----------------------------
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                'Overcast','Sunny','Sunny','Rain','Sunny','Overcast',
                'Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal',
                 'Normal','High','Normal','High','Normal','High',
                 'Normal','High'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No',
                    'Yes','No','Yes','Yes','Yes','Yes',
                    'Yes','No']
})

# --- STEP 2: SHOW DATA ON WEB ---
if st.checkbox("Show Training Data"):
    st.table(data)

# -----------------------------
# Logic (Entropy, IG, ID3)
# -----------------------------
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    ent = 0
    for count in counts:
        p = count / len(col)
        ent -= p * math.log2(p)
    return ent

def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[attribute], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = df[df[attribute] == values[i]]
        weighted_entropy += (counts[i]/len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(df, target, attributes):
    if len(np.unique(df[target])) == 1:
        return df[target].iloc[0]
    if len(attributes) == 0:
        return df[target].mode()[0]
    gains = [information_gain(df, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]
    tree = {best_attr: {}}
    for value in np.unique(df[best_attr]):
        subset = df[df[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = id3(subset, target, remaining_attrs)
    return tree

# -----------------------------
# Build Tree
# -----------------------------
attributes = list(data.columns)
attributes.remove('PlayTennis')
decision_tree = id3(data, 'PlayTennis', attributes)

# --- STEP 3: REPLACE PRINT WITH ST.JSON ---
st.subheader("Generated Decision Tree:")
st.json(decision_tree)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample[attr]
    if value in tree[attr]:
        return predict(tree[attr][value], sample)
    else:
        return "Unknown"

# --- STEP 4: INTERACTIVE INPUT ---
st.divider()
st.subheader("Make a Prediction")
outlook_input = st.selectbox("Select Outlook", ['Sunny', 'Overcast', 'Rain'])
humidity_input = st.selectbox("Select Humidity", ['High', 'Normal'])

sample = {'Outlook': outlook_input, 'Humidity': humidity_input}
result = predict(decision_tree, sample)

# --- STEP 5: REPLACE PRINT WITH ST.SUCCESS ---
if st.button("Predict"):
    st.write(f"Results for: {sample}")
    if result == "Yes":
        st.success(f"PlayTennis: **{result}** ✅")
    else:
        st.error(f"PlayTennis: **{result}** ❌")
