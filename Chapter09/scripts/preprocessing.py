import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    csv = "management_experience_and_salary.csv"
    filename = f"{base_dir}/input/{csv}"
    df_all_data = pd.read_csv(filename)

    X = df_all_data['management_experience_months'].values 
    y = df_all_data['monthly_salary'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, random_state=0
    )

    df_training_data = pd.DataFrame({ 
        'monthly_salary': y_train, 
        'management_experience_months': X_train
    })
    
    csv = "training_data.csv"
    output_path = f"{base_dir}/output/{csv}"

    df_training_data.to_csv(
        output_path, 
        header=False, index=False
    )