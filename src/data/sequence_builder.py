\"\"\"sequence_builder.py - build rolling sequences\"\"\"
import numpy as np

def build_sequences(arr, seq_len=30):
    X=[]
    for i in range(len(arr)-seq_len+1):
        X.append(arr[i:i+seq_len])
    return np.array(X)

if __name__==\"__main__\":
    print(\"Sequence builder stub\")
