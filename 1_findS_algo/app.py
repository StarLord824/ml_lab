import pandas as pd

file_path = "./data.csv"

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)
    
    # Separate the features (X) and the target labels (y)
    X = data.iloc[:, :-1].values  # All columns except the last one (features)
    y = data.iloc[:, -1].values   # Last column (target label)
    
    hypothesis = list(X[0]) 
    
    for i in range(1, len(y)):
        if y[i] == 'Yes':  # Positive example
            for j in range(len(hypothesis)):
                # If the attribute in the hypothesis doesn't match, generalize it
                if hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    
    return hypothesis


most_specific_hypothesis = find_s_algorithm(file_path)
print("Most Specific Hypothesis:", most_specific_hypothesis)
