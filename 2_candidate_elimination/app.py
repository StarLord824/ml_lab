import pandas as pd

# Function to generalize the hypothesis in the S set
def generalize(hypothesis, example):
    return [h if h == e else '?' for h, e in zip(hypothesis, example)]

# Function to specialize the hypothesis in the G set
def specialize(hypothesis, example, attributes):
    new_hypotheses = []
    for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
            continue
        elif hypothesis[i] != example[i]:
            new_hypothesis = hypothesis.copy()
            new_hypothesis[i] = example[i]
            new_hypotheses.append(new_hypothesis)
    return new_hypotheses

def candidate_elimination(file_path):
    # Step 1: Read the training data from a CSV file
    data = pd.read_csv(file_path)
    
    # Separate the features (X) and the target labels (y)
    X = data.iloc[:, :-1].values  # All columns except the last one (features)
    y = data.iloc[:, -1].values   # Last column (target label)
    
    # Step 2: Initialize the General (G) and Specific (S) sets
    attributes = data.columns[:-1]  # List of attribute names
    G = [['?' for _ in attributes]]  # Most general hypothesis
    S = [['0' for _ in attributes]]  # Most specific hypothesis
    
    # Step 3: Iterate through the training examples
    for i in range(len(y)):
        example = X[i]
        label = y[i]
        
        if label == 'Yes':  # Positive example
            # Generalize the hypotheses in S to include the positive example
            S_new = []
            for hypothesis in S:
                generalized_hypothesis = generalize(hypothesis, example)
                if generalized_hypothesis not in S_new:
                    S_new.append(generalized_hypothesis)
            S = S_new
            
            # Remove hypotheses in G that are inconsistent with the positive example
            G_new = []
            for hypothesis in G:
                if all(g == '?' or g == e for g, e in zip(hypothesis, example)):
                    G_new.append(hypothesis)
            G = G_new
        
        elif label == 'No':  # Negative example
            # Remove hypotheses from G that are consistent with the negative example
            G_new = []
            for hypothesis in G:
                if not all(g == '?' or g == e for g, e in zip(hypothesis, example)):
                    G_new.append(hypothesis)
            G = G_new
            
            # Specialize the hypotheses in S that are inconsistent with the negative example
            S_new = []
            for hypothesis in S:
                specialized_hypotheses = specialize(hypothesis, example, attributes)
                S_new.extend(specialized_hypotheses)
            S = S_new

    # Step 4: Output the final hypotheses in G and S
    return G, S

# Path to your CSV file
file_path = "2_candidate_elimination/data.csv"

# Call the Candidate-Elimination algorithm and display the result
G, S = candidate_elimination(file_path)
print("General Hypotheses (G):", G)
print("Specific Hypotheses (S):", S)
