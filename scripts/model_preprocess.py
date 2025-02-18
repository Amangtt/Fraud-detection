import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
def preprocess(input_path,credit_input_path,output_path,credit_output_path,scaler_path):
    m=pd.read_csv(input_path)
    credit=pd.read_csv(credit_input_path)
    m['signup_time'] = pd.to_datetime(m['signup_time'])
    m['purchase_time'] = pd.to_datetime(m['purchase_time'])
    m['hour_of_day'] = m['purchase_time'].dt.hour
    m['day_of_week'] = m['purchase_time'].dt.dayofweek
    m['purchase_delay'] = (m['purchase_time'] - m['signup_time']).dt.total_seconds() / 3600
    user_freq = m.groupby('user_id').size()
    m['user_transaction_frequency'] = m['user_id'].map(user_freq)

    device_freq = m.groupby('device_id').size()
    m['device_transaction_frequency'] = m['device_id'].map(device_freq)

    m['user_transaction_velocity'] = m['user_transaction_frequency'] / m['purchase_delay']
    
    cols=['upper_bound_ip_address','signup_time','purchase_time','lower_bound_ip_address']
    m=m.drop(columns=cols) 
    cols=['user_id','device_id','ip_address','country']
    m=m.drop(columns=cols)
    m= pd.get_dummies(m, columns=['sex','browser','source'], drop_first=True)
    
    numeric_col=['purchase_value', 'age', 'hour_of_day', 'day_of_week',
       'purchase_delay', 'user_transaction_frequency',
       'device_transaction_frequency', 'user_transaction_velocity']
    scaler= StandardScaler()
    m[numeric_col]=scaler.fit_transform(m[numeric_col])
        # Save the scaler
    joblib.dump(scaler, scaler_path)
    m.to_csv(output_path,index=False)
    #credit card part
    credit=credit.drop(columns='Time')
    col=['Amount']
    credit[col]=scaler.fit_transform(credit[col])
    credit.to_csv(credit_output_path,index=False)

preprocess('/Users/amantebeje/Desktop/projects/FRAUD-DETECTION/Data/preprocessed/merged_data.csv','/Users/amantebeje/Desktop/projects/FRAUD-DETECTION/Data/raw/creditcard_edited.csv','/Users/amantebeje/Desktop/projects/FRAUD-DETECTION/Data/preprocessed/final_fraud1.csv','/Users/amantebeje/Desktop/projects/FRAUD-DETECTION/Data/preprocessed/final_credit.csv','/Users/amantebeje/Desktop/projects/FRAUD-DETECTION/model/scaler.pkl')

if __name__ == "__main__" :
    parser= argparse.ArgumentParser()    
    parser.add_argument("--input", required=True, help='Path to read csv')
    parser.add_argument('--output', required=True, help='Path to output csv')
    args=parser.parse_args()
    preprocess(args.input,args.output)
 


