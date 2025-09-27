# import standard library for data cleaning 
import pandas as pd 

# function load_data() takes file path and return a DataFrame
def load_data(path):
    return pd.read_csv(path)

# function clean_data() performs the cleaning steps in two ways 
def clean_data(df):
    # drops any row that contains at least one missing value(NaN)
    df = df.dropna()
    # converts categorical columns into one-hot variables (0/1 numerical columns)
    df = pd.get_dummies(df, drop_first = True)
    return df
# it ensures the codes in the block runs only when the script is excuted directly.
if __name__ == "__main__":
    df = load_data("C:/Users/HP/Desktop/student-performance-analysis/data/raw/student-mat.csv")
    df_clean = clean_data(df)
    df_clean.to_csv("C:/Users/HP/Desktop/student-performance-analysis/data/processed/student_mat_clean.csv", index= False)
    print("student_mat_data cleaned and saved to data/processed")

if __name__ == "__main__":
    df = load_data("C:/Users/HP/Desktop/student-performance-analysis/data/raw/student-por.csv")
    df_clean = clean_data(df)
    df_clean.to_csv("C:/Users/HP/Desktop/student-performance-analysis/data/processed/student_por_clean.csv", index= False)
    print("student_por_data cleaned and saved to data/processed")