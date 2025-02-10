import pandas as pd


class cleaning:
    #loadind data
    def load_data(inputfile1,inputfile2,inputfile3):
        df1=pd.read_csv(inputfile1)
        df2=pd.read_csv(inputfile2)
        df3=pd.read_csv(inputfile3)
        return df1,df2,df3

    def missing_values(c,f,i):
        #Checking for missing values
        print(f'Creditcard dataset{c.isnull().sum()}')
        print(f'Fraud dataset{f.isnull().sum()}')
        print(f'ip dataset{i.isnull().sum()}')

    def dtype(c,f,i):
        #Check datatypes of the dataframes
        print(c.dtypes)
        print(f.dtypes)
        print(i.dtypes)
   

    def check_dup(c,f,i):
        #Checking for duplicates
        print(c.duplicated().sum())
        print(f.duplicated().sum())
        print(i.duplicated().sum())
        c_edited=c.drop_duplicates()
        c_edited.to_csv('./Data/preprocessed/creditcard_edited.csv', index=False)
        return c_edited
   

    def merge(f,i):
        f['ip_address'] = f['ip_address'].astype(int)
        # Convert lower and upper bounds to integers
        i['lower_bound_ip_address'] = i['lower_bound_ip_address'].astype(int)
        i['upper_bound_ip_address'] = i['upper_bound_ip_address'].astype(int)

    # Initialize a list to collect merged rows
        merged_rows = []

        # Iterate through each row in fraud_data and check if ip_address falls within bounds
        for index, row in f.iterrows():
            ip_address = row['ip_address']
            # Find matching rows in ip_data
            matches = i[
                (ip_address >= i['lower_bound_ip_address']) &
                (ip_address <= i['upper_bound_ip_address'])
            ]
            
            # If matches are found, combine them with the current fraud_data row
            if not matches.empty:
                for _, match_row in matches.iterrows():
                    combined_row = pd.concat([row, match_row])
                    merged_rows.append(combined_row)

        # Create a DataFrame from the list of merged rows
        merged_data = pd.DataFrame(merged_rows)

        # Display the merged data
        merged_data.to_csv('./Data/preprocessed/merged_data.csv', index=False)
        return merged_data
  


    