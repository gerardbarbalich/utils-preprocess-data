from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from typing import Dict, List, Any


def create_dict(processing: bool) -> Dict[str, Any]:
    if processing:
        return {
            'df_binary': pd.DataFrame(),
            'df_binary_trimmed': pd.DataFrame(),
            'df_encoded': pd.DataFrame(),
            'df_encoded_trimmed': pd.DataFrame(),
            'df_scaled': pd.DataFrame(),
            'df_scaled_trimmed': pd.DataFrame(),
        }
    else:
        return {
            'df_config': pd.DataFrame(),
            'df_raw': pd.DataFrame(),
            'df_trimmed': pd.DataFrame(),
            'df_processed': pd.DataFrame()
        }




def load_config_file(
    path:str):
    try:
        # Read the config file (assuming it's in CSV format)
        config_data = pd.read_csv(path)
        
        # Optionally, perform any necessary validation or preprocessing of the config_data
        return config_data
    
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Config file at {path} is empty")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse config file at {path}")
        return None


def confirm_dtypes(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    for index, row in config.iterrows():
        column_name = row['column_name']
        dtype = row['dtype']
        
        # Check if the column exists in the dataframe
        if column_name in df.columns:
            column_data = df[column_name]
            
            # Ensure column_data is a Series
            if isinstance(column_data, pd.Series):
                # Convert column to specified dtype if needed
                if dtype == 'int':
                    df[column_name] = pd.to_numeric(column_data, errors='coerce').astype('Int64')
                elif dtype == 'float':
                    df[column_name] = pd.to_numeric(column_data, errors='coerce')
                elif dtype == 'bool':
                    df[column_name] = column_data.astype(bool)
                elif dtype == 'date':
                    df[column_name] = pd.to_datetime(column_data, errors='coerce')
                elif dtype == 'datetime':
                    df[column_name] = pd.to_datetime(column_data, errors='coerce')
                elif dtype == 'categorical':
                    df[column_name] = column_data.astype('category')
                # Handle other data types if needed
            else:
                print(f"Column {column_name} is not a Series.")
            
    return df


# Function to check and coerce data types
def check_data_types(
    df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    try:
        # Iterate through each row in the config
        for index, row in config.iterrows():
            column_name = row['column_name']
            desired_dtype = row['dtype']
            
            # Check if column exists in the DataFrame
            if column_name in df.columns:
                current_dtype = df[column_name].dtype
                
                # Coerce the data type if it doesn't match the desired type
                if current_dtype != desired_dtype:
                    # Coercion logic based on desired_dtype
                    # For example:
                    if desired_dtype == 'int':
                        df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(int)
                    elif desired_dtype == 'float':
                        df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(float)
                    # Add conditions for other data types as needed
                    
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def find_correlation(
    corr, cutoff=0.7, exact=None):
    """
    This function is the Python implementation of the R function 
    `findCorrelation()`. It searches through a correlation matrix and returns a list of column names 
    to remove to reduce pairwise correlations.
    
    For the documentation of the R function, see 
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `findCorrelation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R
    
    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be 
        recomputed at each step
    """
    
    def _findCorrelation_fast(
        corr, avg, cutoff):

        combsAboveCutoff = corr.where(lambda x: (np.tril(x)==0) & (x > cutoff)).stack().index

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol


    def _findCorrelation_exact(
        corr, avg, cutoff):

        x = corr.loc[(*[avg.sort_values(ascending=False).index]*2,)]

        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

        x.values[(*[np.arange(len(x))]*2,)] = np.nan

        deletecol = []
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix+1:]:
                if x.loc[i, j] > cutoff:
                    if x[i].mean() > x[j].mean():
                        deletecol.append(i)
                        x.loc[i] = x[i] = np.nan
                    else:
                        deletecol.append(j)
                        x.loc[j] = x[j] = np.nan
        return deletecol

    
    if not np.allclose(corr, corr.T) or any(corr.columns!=corr.index):
        raise ValueError("correlation matrix is not symmetric.")
        
    acorr = corr.abs()
    avg = acorr.mean()
        
    if exact or exact is None and corr.shape[1]<100:
        return _findCorrelation_exact(acorr, avg, cutoff)
    else:
        return _findCorrelation_fast(acorr, avg, cutoff)
    
    
def assess_correlation_trim_columns(
    df: pd.DataFrame, cutoff=0.7):
    try:
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Plot correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
        
        # Find highly correlated columns
        columns_to_drop = find_correlation(correlation_matrix, cutoff)
        
        # Drop highly correlated columns
        df = df.drop(columns=columns_to_drop)
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
def scale_features(
    df: pd.DataFrame, list_of_columns: list) -> pd.DataFrame:
    scaled_df = StandardScaler().fit_transform(df[list_of_columns])
    
    # Ensure the shape of scaled_df aligns with the intended columns
    if len(list_of_columns) != scaled_df.shape[1]:
        print("The number of columns doesn't match the shape of scaled_df")
        return None, None
    
    scaled_and_labelled_df = pd.DataFrame(scaled_df, columns=list_of_columns, index=df.index)
    return scaled_and_labelled_df


def scale_and_assess_feature_correlation(
    df: pd.DataFrame, list_of_columns: list) -> pd.DataFrame:
    """Once scaled, this assesses the correlation of the features"""
    scaled_and_labelled_df = scale_features(df, list_of_columns)
    correlation_matrix = scaled_and_labelled_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    return scaled_and_labelled_df


def get_column_names_by_dtype(
    config: pd.DataFrame, includes: list):
    ls_filtered_cols = [new if isinstance(new, str) else old for new, old in zip(config.loc[config['dtype'].isin(includes), 'column_name_new'], config.loc[config['dtype'].isin(includes), 'column_name'])]
    return ls_filtered_cols


def process_data_according_to_config(
    config: pd.DataFrame, data: pd.DataFrame):
    
    process_data = create_dict(processing=True)
    
    # Filter the DataFrame based on dtype == 'bool' and retrieve the column names
    # Combine 'column_name' and 'column_name_new' values into a single list
    ls_bool_columns = get_column_names_by_dtype(config, ['bool'])
    process_data['df_binary'] = data['df_trimmed'][ls_bool_columns]

    # Replace 'Yes'/'No' strings with 1/0 respectively in the specified columns
    process_data['df_binary'] = process_data['df_binary'].applymap(lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else x))

    # Replace True/False with 1/0 respectively in the specified columns
    process_data['df_binary'] = process_data['df_binary'] .replace({True: 1, False: 0})

    
    # Encode categorical data
    ls_cat_columns = get_column_names_by_dtype(config, ['categorical'])
    encoder = OneHotEncoder(sparse=False, dtype=int)
    encoded_data = encoder.fit_transform(data['df_trimmed'][ls_cat_columns])
    process_data['df_encoded'] = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(ls_cat_columns))
    process_data['df_encoded'].index = data['df_trimmed'][ls_cat_columns].index 
    
    # Scale numeric data
    ls_numeric_cols = get_column_names_by_dtype(config, ['int', 'float'])
    process_data['df_scaled'] = scale_and_assess_feature_correlation(data['df_trimmed'], ls_numeric_cols) # none of these are over 0.8 correlation 
    
    # Section 3: Assess features, trim, and concat all dataframes
    process_data['df_binary_trimmed'] = process_data['df_binary'].copy()
    # process_data['df_binary_trimmed'] = assess_correlation_trim_columns(process_data['df_binary']) # TO DO - this drops Churn and SeniorCitizen
    process_data['df_scaled_trimmed'] = assess_correlation_trim_columns(process_data['df_scaled'])
    process_data['df_encoded_trimmed'] = assess_correlation_trim_columns(process_data['df_encoded'])

    # Concat all of the dfs
    df = pd.concat(
        [process_data['df_binary_trimmed'], 
        process_data['df_scaled_trimmed'],
        process_data['df_encoded_trimmed']], axis=1)#.dropna(axis=0)
    
    # Check for missing data
    print('Missing values:\n', df.isnull().sum())
    
    return df


# Function to export processed data
def export_processed_data(
    df: pd.DataFrame, file_path: str):
    try:
        # Export processed data to a CSV file
        df.to_csv(file_path, index=False)  # Set index=False to exclude row indices
        
        print(f"Processed data has been exported to {file_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        

def export_processed_descriptions(
    df: pd.DataFrame, file_path: str):
    try:
        # Export processed data descriptions to a CSV file
        df.describe().T.reset_index().to_csv(file_path, index=False, header=True) 
                
        print(f"Processed data descriptions have been exported to {file_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        

def read_in_drop_and_rename(
    file_path: str, df_config: pd.DataFrame) -> pd.DataFrame:
    try:
        # Read in the file
        df = pd.read_csv(file_path)
        print(df)
        
        # Drop columns not listed in df_config['column_name']
        columns_to_keep = df_config['column_name'].tolist()
        df = df[columns_to_keep]
        
        # Rename columns if 'column_name_new' is listed in df_config
        rename_dict = dict(zip(df_config['column_name'], df_config['column_name_new']))
        # Remove key-value pairs where value is NaN
        rename_dict = {key: value for key, value in rename_dict.items() if not pd.isnull(value)}
        
        df = df.rename(columns=rename_dict)
   
        
        # Drop any columns not listed in df_config['column_name']
        # df_columns = df.columns.tolist()
        # columns_to_drop = [col for col in df_columns if col not in columns_to_keep]
        # df = df.drop(columns=columns_to_drop)
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Main function to orchestrate the process
def main():
    config_path = 'data/raw/config-preprocess-data.csv'
    output_path = 'data/processed/processed_data.csv'
    output_describe_path = 'data/processed/processed_data_describe.csv'
    
    # Init a blank dict
    data = create_dict(processing=False)
    
    # Load in config file
    data['df_config'] = load_config_file(config_path)
    
    # Process each file based on config
    # for file_name in config_data['file_name'].unique():
    for file_name in data['df_config']['file_name'].unique():
        file_path = f'data/raw/{file_name}'
                
        # Import data, rename columns, confirm dtypes, and drop unnneded columns
        if data['df_config'] is not None:
            data['df_trimmed'] = read_in_drop_and_rename(file_path, data['df_config'])
            data['df_trimmed'] = confirm_dtypes(data['df_trimmed'].copy(), data['df_config'])
            
            # Process the data
            data['df_processed'] = process_data_according_to_config(config=data['df_config'], data=data)
        
        # Export processed data and summary of the numerical columns
        if data['df_processed'] is not None:  
            export_processed_data(data['df_processed'], output_path)
            export_processed_descriptions(data['df_trimmed'], output_describe_path)
    

if __name__ == "__main__":
    main()