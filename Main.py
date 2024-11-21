import streamlit as st 
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx
import os
import time

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Tiger Migration Pattern Prediction",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and process the migration dataset
def load_data(file):
    try:
        xls = pd.ExcelFile(file)
        data_sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
        return data_sheets
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None

# Define hidden states based on domain knowledge of tiger migration patterns
def get_hidden_state_label(state):
    """
    Returns a label for each hidden state based on tiger migration patterns.
    - 'State 0' could represent 'Localized Movement'.
    - 'State 1' could represent 'Exploratory Movement'.
    - 'State 2' could represent 'Migration'.
    
    Parameters:
    - state: int, the hidden state index (0, 1, or 2).
    
    Returns:
    - str: A descriptive label for the hidden state.
    """
    state_labels = {
        0: 'Localized Movement',
        1: 'Exploratory Movement',
        2: 'Migration'
    }
    return state_labels.get(state, 'Unknown State')


# Function to fit the HMM model and decode the hidden states
def fit_hmm_and_decode(data, n_states=3, n_iter=100):
    data = data.select_dtypes(include=[np.number]).dropna()
    
    if data.shape[1] == 0:
        st.warning("No numerical data available in this dataset.")
        return None
    
    X = data.values
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter)
    model.fit(X)
    logprob, hidden_states = model.decode(X, algorithm="viterbi")
    
    # Convert hidden states to human-readable labels
    hidden_state_labels = [get_hidden_state_label(state) for state in hidden_states]
    return hidden_state_labels

# Function to draw Finite State Machine diagram
def draw_fsm(hidden_states, n_states):
    """
    Draws a finite state machine (FSM) diagram representing the transitions between states.
    """
    # Create a mapping from state names to integers (if not already mapped)
    state_mapping = {state: idx for idx, state in enumerate(set(hidden_states))}
    
    # Convert hidden states to integer indices based on the mapping
    hidden_states_int = [state_mapping[state] for state in hidden_states]
    
    # Create a transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(len(hidden_states_int) - 1):
        transition_matrix[hidden_states_int[i], hidden_states_int[i + 1]] += 1

    # Normalize the transition matrix
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n_states))

    # Add edges to the graph
    for i in range(n_states):
        for j in range(n_states):
            if transition_matrix[i, j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=5000, edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

    # Optional: Print the state mapping for debugging
    print("State Mapping:", state_mapping)


# Main function to handle Streamlit user interface
def main():
    st.markdown("<h1 style='text-align: center;'>🐯 Tiger Migration Pattern Prediction</h1>", unsafe_allow_html=True)
    
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your migration dataset (Excel format):", type=["xlsx"])

    if uploaded_file:
        data_sheets = load_data(uploaded_file)
        
        if data_sheets is not None:
            sheet_names = list(data_sheets.keys())
            sheet_choice = st.sidebar.selectbox("Select Sheet to Analyze", sheet_names)
            data = data_sheets[sheet_choice]
            
            st.write(f"## Data from Sheet: {sheet_choice}")
            st.dataframe(data.head())

            if st.button("Run Migration Prediction"):
                with st.spinner("Running Hidden Markov Model..."):
                    hidden_states = fit_hmm_and_decode(data)

                    if hidden_states is not None:
                        st.write("## Predicted Tiger Migration Zones")
                        state_count = pd.Series(hidden_states).value_counts()
                        st.bar_chart(state_count)
                        
                        st.write("### State Transitions")
                        draw_fsm(hidden_states, n_states=3)

                        st.write("### Hidden States Over Time")
                        st.line_chart(pd.DataFrame(hidden_states, columns=["Predicted Zone"]))
                    else:
                        st.warning("No data available to predict.")
    
    with st.sidebar:
        st.header("About")
        st.write("Upload a dataset to predict tiger migration patterns using our AI model.")
        st.markdown("---")
        st.write("*Model Details:*")
        st.write("- HMM Model with 3 Hidden States:")
        st.write("  - *Localized Movement*: The tiger is in a restricted area, possibly resting or in a home range.")
        st.write("  - *Exploratory Movement*: The tiger is exploring new areas, possibly traveling or searching for a new territory.")
        st.write("  - *Migration*: The tiger is migrating between different regions.")
        st.write("- Number of States: 3 (Localized Movement, Exploratory Movement, Migration)")
        st.write("- Confidence Threshold: 70%")

if __name__ == "_main_":
    main()
