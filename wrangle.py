import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from scipy import stats

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


from IPython.display import display, Markdown

def load_csv_file(filename):
    try:
        if not os.path.exists(filename):
            print(f"The file: {filename} doesn't exist")
        else:
            print("Found File")
            return pd.concat([chunk for chunk in tqdm(pd.read_csv(filename, chunksize=1000), desc=f'Loading {filename}')])
    except:
        print("Didn't Work! :(")

def rename_columns(df):
    new_names = []

    for column in df.columns:
        level_one = re.sub('(?<!^)(?=[A-Z])', '_', column).lower()
        level_one = re.sub(' ', '_',level_one)
        level_one = re.sub(' _', '_',level_one)
        level_one = re.sub('__','_',level_one)
        new_names.append(level_one)
    df.columns = new_names
    return df

def data_dictionary(df):
    # Printing a data dictionary using a printout of each column name
    # formatted as a MarkDown table
    # =================================================================

    # variable to hold size of longest string in dataframe column names
    size_longest_name = len((max((df.columns.to_list()), key = len)))

    # head of markdown table
    print(f"| {'Name' : <{size_longest_name}} | Definition |")
    print(f"| {'-'*size_longest_name} | {'-'*len('Definition')} |")

    # dataframe column content
    for i in (df.columns.to_list()):
        print(f"| {i : <{size_longest_name}} | Definition |")

def plot_clust(df):

    num, cat = separate_column_type_list(df)
    train_scaled = df[num]
    
    # Create Object
    mm_scaler = MinMaxScaler()
    train_scaled[num] = mm_scaler.fit_transform(train_scaled[num])
    seed = 42
    cluster_count = 4

    kmeans = KMeans(n_clusters=cluster_count,random_state=seed)
    kmeans.fit(train_scaled)
    df['clusters']=kmeans.predict(train_scaled)
    sns.boxplot(data=df,x='clusters',y='alcohol',hue='quality')
    plt.title("What about Clustering?")
    plt.show()

def flatten(list_nd):
    '''
    Function that allows other functions to accept a list as an argument
    and not cause pesky 2d-List issues when appending a list to the 
    original argument list
    '''
    flattened = []
    for item in list_nd:
        if isinstance(item,list): flattened.extend(flatten(item))
        else: flattened.append(item)
    return flattened

def split_data(df):
    '''
    This function take in a dataframe and splits into train validate test
    '''
    
    # create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    
    # create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)
    
    # sanity check
    print(train.shape,validate.shape,test.shape)
    
    return train, validate, test

def isolate_target(df, target):
    '''
    splits datasets into X,y
    '''
    
    #Split into X and y
    X = df.drop(columns=[target])
    y = df[target]

    # sanity check
    print(X.shape,y.shape)
    
    return X,y

def dummies(df,dummies):
    # keeper columns are numerical & discrete chosen
    # for dummy creation
    numerical = df.select_dtypes('number').columns
    keepers = df[numerical].columns.to_list()
    keepers.append(dummies)
    
    # fix list to be useable as column index
    keepers = flatten(keepers)
    
    # Create dummies for non-binary categorical columns
    df[dummies]=pd.get_dummies(df[dummies], drop_first = True)
    
    # drop redundant column
    df = df.drop(df.columns.difference(keepers),axis=1)
    
    return df

def eval_results(p, alpha, group1, group2):
    '''
        Test Hypothesis  using Statistics Test Output.
        This function will take in the p-value, alpha, and a name for the 2 variables
        you are comparing (group1 and group2) and return a string stating 
        whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Reject $H_0$"))
        display(Markdown( f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'))
    
    else:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Failed to Reject $H_0$"))
        display(Markdown( f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'))
        
def separate_column_type_list(df):
    '''
        Creates 2 lists separating continous & discrete
        variables.
        
        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame from which columns will be sorted.
        
        Returns
        ----------
        continuous_columns : list
            Columns in DataFrame with numerical values.
        discrete_columns : list
            Columns in DataFrame with categorical values.
    '''
    continuous_columns = []
    discrete_columns = []
    
    for column in df.columns:
        if (df[column].dtype == 'int' or df[column].dtype == 'float') and ('id' not in column) and (df[column].nunique()>10):
            continuous_columns.append(column)
        elif(df[column].dtype == 'int' or df[column].dtype == 'float') and (df[column].nunique()>11):
            continuous_columns.append(column)
        else:
            discrete_columns.append(column)
            
    return continuous_columns, discrete_columns

def scale_data(df,mode="minmax"):
    # create a list of only continous features from input DataFrame
    continous = df.select_dtypes('number').columns
    
    if mode == "minmax":
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(df[continous])
        df[continous] = scaler.transform(df[continous])
        
        return df

    elif mode == "standard":
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(df[continous])
        df[continous] = scaler.transform(df[continous])
        
        return df
    
    else:
        print("write new code")

def num_vs_num_visualize(df,feature,target):
    question = f"Does a higher {feature} mean higher {target}?"

    #sns.scatterplot(x=df[feature], y=df[target])
    #plt.suptitle(f"{question}")

    #plt.show()
    
    sns.jointplot(data=df, x=feature, y=target,  kind='reg', height=8)
    plt.show()

def eval_results(p, alpha, group1, group2):
    '''
        Test Hypothesis  using Statistics Test Output.
        This function will take in the p-value, alpha, and a name for the 2 variables
        you are comparing (group1 and group2) and return a string stating 
        whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Reject $H_0$"))
        display(Markdown( f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'))
    
    else:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Failed to Reject $H_0$"))
        display(Markdown( f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'))

def question_hypothesis_test(question_number,df,feature,target,alpha=.05):
    num, cat = separate_column_type_list(df)
    question = 'temp'
    if (target in cat) and (feature in num):
        # calculation
        overall_alcohol_mean = df[feature].mean()
        quality_sample = df[df[target] >= 7][target]
        t, p = stats.ttest_1samp(quality_sample, overall_alcohol_mean)
        value = t
        p_value = p/2
        
        # Output variables
        test = "1-Sample T-Test"

        # Markdown Format Question
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        
        # Visualize Question

        # Markdown Formatting Metrics
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{feature}` and `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{feature}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        # Evaluate Results
        eval_results(p_value, alpha, feature, target)

    elif (target in cat) and (feature in cat):
        # calculations
        observed = pd.crosstab(df[feature], df[target])
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        value = chi2
        p_value = p
        
        # Output variables
        test = "Chi-Square"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{feature}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{feature}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        eval_results(p_value, alpha, feature, target)

    elif (target in num) and (feature in num):
        # Markdown Format Question
        question = f"Does a higher {feature} mean higher {target}?"
    
        # calculations
        r, p = stats.pearsonr(df[feature], df[target])
        value = r
        p_value = p
        
        # Output variables
        test = "Pearson's R"

        # Output Question
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        
        # Visualize Question
        num_vs_num_visualize(df,feature,target)
        
        # Markdown Format Metrics
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{feature}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{feature}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))
    
        # Markdown Format Evaluate Results
        eval_results(p_value, alpha, feature, target)
        display(Markdown("<hr style=\"border:2px solid gray\">"))
    else:
        print("write code for different test")

def model():
    print("You write model code now!")

def clean(hot_100,spotify_popular):
    # date type column
    hot_100['chart_date']=pd.to_datetime(hot_100['chart_date'])

    # set date column as index
    hot_100 = hot_100.set_index("chart_date").sort_index()
    hot_100_2000_2020 = hot_100.loc['2000-01-01':'2020-12-31']

    # create song index
    hot_100_2000_2020["song_index"] = hot_100_2000_2020["performer"].str.lower() + " - " + hot_100_2000_2020["song"].str.lower()

    #saving extra copy of date before grouping by index
    hot_100_2000_2020['date'] = hot_100_2000_2020.index

    # Removing Song instance for other weeks besides the most recent
    hot_100_2000_2020_only_latest = hot_100_2000_2020.groupby('song_index', group_keys=False).apply(lambda x: x.index[np.argmax(x.index)])

    # converting back to dataframe
    hot_100_2000_2020_only_latest = pd.DataFrame(hot_100_2000_2020_only_latest)

    # swapping back the index
    hot_100_2000_2020_only_latest.columns = ["index"]

    hot_100_2000_2020_only_latest["unique"]=hot_100_2000_2020_only_latest.index

    hot_100_2000_2020_only_latest = hot_100_2000_2020_only_latest.set_index("index").sort_index()
    hot_100_2000_2020_only_latest['is_unique'] = 1
    hot_100_2000_2020_only_latest['date']=hot_100_2000_2020_only_latest.index
    hot_100_2000_2020_only_latest['song_index']=hot_100_2000_2020_only_latest['unique']
    hot_100_2000_2020_only_latest = hot_100_2000_2020_only_latest.drop(columns = ['unique'])
    merged_hot100_only_latest = hot_100_2000_2020.merge(hot_100_2000_2020_only_latest,how='left')
    merged_hot100_only_latest=merged_hot100_only_latest[merged_hot100_only_latest['is_unique']==1]

    # more
    merged_hot100_only_latest['1st_split']=merged_hot100_only_latest['performer'].str.split('&').str[0]
    merged_hot100_only_latest['2nd_split']=merged_hot100_only_latest['1st_split'].str.split('With').str[0]
    merged_hot100_only_latest['3rd_split']=merged_hot100_only_latest['2nd_split'].str.split('Featuring').str[0]
    merged_hot100_only_latest['4th_split']=merged_hot100_only_latest['3rd_split'].str.split(',').str[0]
    merged_hot100_only_latest['5th_split']=merged_hot100_only_latest['4th_split'].str.split(' x ').str[0]
    merged_hot100_only_latest['6th_split']=merged_hot100_only_latest['5th_split'].str.split('+').str[0]
    # final split on peformer column is aved as "singer" column to use in Unique ID creation
    merged_hot100_only_latest['singer']=merged_hot100_only_latest['6th_split']
    # drop old columns used as step/place-holders to isolate Artist/Singer from "Performer" column
    merged_hot100_only_latest = merged_hot100_only_latest.drop(columns = ['1st_split','2nd_split','3rd_split','4th_split','5th_split','6th_split'])
    # Fixing Song Title mixed characters
    merged_hot100_only_latest['song'] = merged_hot100_only_latest['song'].str.replace('[^0-9a-z - A-Z]', '')
    # Unique ID from artist & song combination
    merged_hot100_only_latest['song_index'] = merged_hot100_only_latest['singer'].str.lower() + ' - ' + merged_hot100_only_latest['song'].str.lower()
    # standardization for both datasets song_index
    merged_hot100_only_latest['song_index'] = merged_hot100_only_latest['song_index'].str.replace('[^0-9a-z-A-Z]', '')
    merged_hot100_only_latest['song_index'] = merged_hot100_only_latest['song_index'].str.replace('*', '')
    merged_hot100_only_latest['song_index'] = merged_hot100_only_latest['song_index'].str.replace('-', ' - ')

    spotify_popular['1st_split']=spotify_popular['artist'].str.split('&').str[0]
    spotify_popular['2nd_split']=spotify_popular['1st_split'].str.split('With').str[0]
    spotify_popular['3rd_split']=spotify_popular['2nd_split'].str.split('Featuring').str[0]
    spotify_popular['4th_split']=spotify_popular['3rd_split'].str.split(',').str[0]
    spotify_popular['performer']=spotify_popular['4th_split']
    spotify_popular = spotify_popular.drop(columns = ['1st_split','2nd_split','3rd_split','4th_split'])
    spotify_popular['song'] = spotify_popular['song'].str.replace('[^0-9a-z - A-Z]', '')
    spotify_popular['song_index'] = spotify_popular['performer'].str.lower() + ' - ' + spotify_popular['song'].str.lower()
    spotify_popular['song_index'] = spotify_popular['song_index'].str.replace('[^0-9a-z-A-Z]', '')
    spotify_popular['song_index'] = spotify_popular['song_index'].str.replace('*', '')
    spotify_popular['song_index'] = spotify_popular['song_index'].str.replace('-', ' - ')

    # M E R G E
    new_spotify_merge = spotify_popular.merge(merged_hot100_only_latest,how='left',on='song_index')

    # N U L L S
    merged_data_non_nulls = new_spotify_merge.dropna()
    merged_data_non_nulls.rename(columns = {'song_x':'song'}, inplace = True)
    merged_data_non_nulls.columns
    merged_data_non_nulls=merged_data_non_nulls.drop(columns=['is_unique','singer'])
    merged_data_non_nulls=merged_data_non_nulls.drop(columns=['performer_x', 'song_index', 'song_y', 'performer_y'])

    # C S V
    merged_data_non_nulls.to_csv('merged_data_non_nulls.csv',index=False)
    
    return merged_data_non_nulls

def acquire():
    hot_100 = pd.read_csv("hot_100.csv")
    spotify_popular = pd.read_csv("songs_normalize.csv")

    df = clean(hot_100,spotify_popular)
    
    return df

def scrub(df):
    
    df = df.drop(columns = ['chart_position', 'instance','popularity',
       'consecutive_weeks', 'previous_week', 'peak_position',
       'worst_position'])
