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
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values  
    
    attributes = data.columns[:-1]  
    G = [['?' for _ in attributes]] 
    S = [['0' for _ in attributes]]  
    
    for i in range(len(y)):
        example = X[i]
        label = y[i]
        
        if label == 'Yes':  
            S_new = []
            for hypothesis in S:
                generalized_hypothesis = generalize(hypothesis, example)
                if generalized_hypothesis not in S_new:
                    S_new.append(generalized_hypothesis)
            S = S_new
            
            G_new = []
            for hypothesis in G:
                if all(g == '?' or g == e for g, e in zip(hypothesis, example)):
                    G_new.append(hypothesis)
            G = G_new
        
        elif label == 'No':  # Negative example
            G_new = []
            for hypothesis in G:
                if not all(g == '?' or g == e for g, e in zip(hypothesis, example)):
                    G_new.append(hypothesis)
            G = G_new
            
            S_new = []
            for hypothesis in S:
                specialized_hypotheses = specialize(hypothesis, example, attributes)
                S_new.extend(specialized_hypotheses)
            S = S_new

    return G, S

file_path = "2_candidate_elimination/data.csv"

G, S = candidate_elimination(file_path)
print("General Hypotheses (G):", G)
print("Specific Hypotheses (S):", S)
