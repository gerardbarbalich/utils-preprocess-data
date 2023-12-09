# Preprocess Data Script

This Python script is designed for preprocessing and cleaning datasets based on a configuration file. It provides functionality for reading, processing, and exporting datasets, with an emphasis on handling various data types.

## Script Overview

### Config File

The config file, `'data/raw/config-preprocess-data.csv'`, should inlclude:

`file_name`: str - string of file name, including `'.csv'`
`column_name`: str - string of existing column name
`column_name_new`: str - string of new column name that will replace `'column_name'` in processing, or `NaN`
`dtype`: str - string of data type (`'bool'`, `'int'`, `'float'`, `'categorical'`)

### Functions

- **`create_dict(processing: bool) -> Dict[str, Any]`**: Creates a dictionary structure to store different dataframes during the preprocessing process.

- **`load_config_file(path: str)`**: Loads and validates a configuration file containing information about columns, datatypes, and processing methods.

- **`confirm_dtypes(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame`**: Confirms and adjusts data types in a dataframe based on the provided configuration.

- **`check_data_types(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame`**: Checks and coerces data types in a dataframe based on the specified configuration.

- **`find_correlation(corr, cutoff=0.7, exact=None)`**: Finds and returns a list of column names to remove to reduce pairwise correlations in a correlation matrix.

- **`assess_correlation_trim_columns(df: pd.DataFrame, cutoff=0.7)`**: Assesses correlation, trims highly correlated columns, and returns the processed dataframe.

- **`scale_features(df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler)`**: Scales features in a dataframe and returns the scaled dataframe and scaler.

- **`scale_and_assess_feature_correlation(df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler)`**: Scales features and assesses their correlation.

- **`process_data_according_to_config(config: pd.DataFrame, data)`**: Processes data based on the provided configuration, including handling boolean values and encoding categorical variables.

- **`export_processed_data(df: pd.DataFrame, file_path: str)`**: Exports the processed dataframe to a CSV file.

- **`export_processed_descriptions(df: pd.DataFrame, file_path: str)`**: Exports the descriptions of the processed dataframe to a CSV file.

- **`read_in_drop_and_rename(file_path: str, df_config: pd.DataFrame) -> pd.DataFrame`**: Reads in a CSV file, drops unnecessary columns, and renames columns based on the provided configuration.

- **`main()`**: Main function orchestrating the preprocessing process.

### Usage

- Ensure the configuration file (`config-preprocess-data.csv`) is available in the 'data/raw/' directory.

- Update the file paths for input and output as needed.

- Run the script to preprocess the data according to the specified configuration.

## Example Usage

```python
python preprocess_data.py
```