
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    #detects skewdness of data  
    def detect_outliers_with_boxplot(df, column):
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column], color="skyblue")
        plt.title(f"Box Plot of {column}", fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.show()
    
    # Accepts dataframe and column for numeric analysis 
    def numerical_univariate(df, col):
            plt.figure(figsize=(15, 5))
            sns.histplot(df[col], bins=10, kde=True)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
 
    #Accepts dataframe and column for catagorical univariate analysis 
    def categorical_univariate(df, col):
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col)
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
  
   #classify frauds based on browser
    def browser(df, col):
        class_1=df[df['class']==1]
        agg_browser=class_1.groupby(col).size().reset_index(name='count')
        plt.title('Mostly used browsers in fraud activity')
        plt.bar(agg_browser[col], agg_browser['count'], color='skyblue', edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


    #classify based on age group
    def age_group(df):
        bins = [0, 18, 30, 50, float('inf')]  # Bins: (0-18], (18-30], (30-50], (50+)
        labels = ['>18', '18-30', '30-50', '50<']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
        # Filter the data for class = 1
        class_1 = df[df['class'] == 1]

        # Group by 'age_group' and count occurrences
        class_1_age_group_counts = class_1.groupby('age_group').size().reset_index(name='count')

        plt.figure(figsize=(8, 8))
        plt.pie(
        class_1_age_group_counts['count'], 
        labels=class_1_age_group_counts['age_group'], 
        autopct='%1.0f%%', 
        startangle=90, 
        colors=plt.cm.Pastel1.colors
    )
        plt.show()
