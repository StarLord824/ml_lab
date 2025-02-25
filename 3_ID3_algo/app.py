import pandas as pd
import math
import os

def entropy(data):
    labels = data.iloc[:, -1]
    total_count = len(labels)
    label_counts = labels.value_counts()
    entropy_value = 0
    for count in label_counts:
        prob = count / total_count
        entropy_value -= prob * math.log2(prob)
    return entropy_value

def information_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

def id3(data, attributes):
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]

    if len(attributes) == 0:
        return data.iloc[:, -1].mode()[0]
    
    gains = {attribute: information_gain(data, attribute) for attribute in attributes}
    best_attribute = max(gains, key=gains.get)
    
    tree = {best_attribute: {}}
    
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        tree[best_attribute][value] = id3(subset, remaining_attributes)
    
    return tree

def predict(tree, example):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        attribute_value = example[attribute]
        return predict(tree[attribute][attribute_value], example)
    else:
        return tree

def validate_and_load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        
        data = pd.read_csv(file_path)
        
        if data.empty:
            raise ValueError(f"The file {file_path} is empty.")
        
        return data
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty or has no valid data.")
        return None
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return None
    except ValueError as val_error:
        print(val_error)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


file_path = "3_ID3_algo/sample.csv" 

data = validate_and_load_data(file_path)

if data is not None:
    attributes = data.columns[:-1]  
    decision_tree = id3(data, attributes)

    print("Decision Tree:")
    print(decision_tree)

    new_sample = {
        'Outlook': 'Sunny',
        'Temperature': 'Hot',
        'Humidity': 'High',
        'Windy': 'FALSE'
    }

    new_sample_series = pd.Series(new_sample)

    prediction = predict(decision_tree, new_sample_series)
    print(f"The predicted class for the new sample is: {prediction}")
