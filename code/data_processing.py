import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


def counter_nan(series):
    return sum(series.isna())


def get_percentage_missing_data(df, threshold, show_percentage=False):
    '''Inputs: Datframe, thresold
    Function: return the features with more than thresold of missing data'''
    rows = df.shape[0]
    missing_features = []
    perc_missing_features = {}
    for column in df.columns:
        counter_perc = counter_nan(df[column]) / rows
        perc_missing_features[column] = round(counter_perc * 100, 2)
        if counter_perc > threshold:
            missing_features.append(column)
        else:
            continue

    if show_percentage:
        print(perc_missing_features)
    else:
        pass
    return missing_features


# Fill nan employment status clients with indefined
def fill_nan_employment(df):
    '''Transforms the employ status clients of the dataframe for non defined'''
    df['employment_status_clients'] = df['employment_status_clients'].replace(to_replace=np.nan, value='non-defined')
    return df


def birthday_to_age(string_date):
    '''Takes the birthdate in a string format and returns the age'''
    #Calculate the age given the birthday (year 2018)
    current_year = 2018
    year_birthday = int(string_date.split('-')[0])
    age = current_year - year_birthday
    return age


def get_interest_data(previous_loan_df):
    '''Input: dataframe with the previous loans data
    Function: add two new columns to the dataframe containing the interest rate and the rate amount'''
    total_due = previous_loan_df['totaldue']
    loan_amount = previous_loan_df['loanamount']
    interest = total_due - loan_amount
    interest_rate = round((interest / loan_amount) * 100, 2)
    previous_loan_df['interest'] = interest
    previous_loan_df['interest_rate'] = interest_rate

    return previous_loan_df


def scale_termdays(previous_loan_df):
    '''Scale termdays using a year as unit'''
    previous_loan_df['termdays'] = round(previous_loan_df['termdays'] / 365, 2)
    return previous_loan_df


def get_date_features(df, keep_customer_id=False):
    '''Return a vector with the name of the date features'''
    if keep_customer_id:
        date_features = ['customerid']
    else:
        date_features = []

    for column in df.columns:
        if 'date' in column:
            date_features.append(column)
        else:
            continue
    return date_features


def get_non_date_features(df, keep_customer_id=False):
    '''Return a vector with the non date features'''
    if keep_customer_id:
        non_date_features = ['customerid']
    else:
        non_date_features = []

    for column in df.columns[1:]:
        if 'date' not in column:
            non_date_features.append(column)
        else:
            continue
    return non_date_features


def string_to_datetime(df):
    '''Convert date with string format into date variables'''
    for date_feature in get_date_features(df):
        df[date_feature] = df[date_feature].apply(lambda date: datetime.strptime(date.split(' ')[0], '%Y-%m-%d'))
    return df


def differentiate_dates(df, columns_to_differentiate=[]):
    '''Create a new column in which we store the distance between two dates'''
    # Days scale
    name_new_column = 'diff_' + columns_to_differentiate[0] + '_' + columns_to_differentiate[1]
    sec_in_day = 60 * 60 * 24
    #
    df[name_new_column] = df[columns_to_differentiate[0]] - df[columns_to_differentiate[1]]
    df[name_new_column] = df[name_new_column].apply(lambda x: round((x.total_seconds() / sec_in_day), 2))

    return df


def date_to_day(df):
    '''Create a new variable containing in which day of the week the due and the repaid took place'''
    columns_map = {
        'firstrepaiddate': 'repaid_dayofweek',
        'firstduedate': 'due_dayofweek'
    }
    for key, value in columns_map.items():
        df[value] = df[key].apply(lambda date: datetime.weekday(date))
    return df


def is_month_start(df):
    columns_map = {
        'firstrepaiddate': 'is_month_start_repaid',
        'firstduedate': 'is_month_start_due'
    }
    for key, value in columns_map.items():
        df[value] = df[key].apply(lambda date: 1 if date.day == 1 else 0)
    return df


def get_statistics_by_customer(df, stats=['max', 'min', 'mean']):
    '''Return a df with the aggregated data by customer
    '''
    df_dict = {}
    for stat in stats:
        if stat == 'max':
            rename_feature = {}
            for column in df.columns[2:]:
                rename_feature[column] = column + '_' + stat
            df_dict[stat] = df.groupby(['customerid']).max()
            df_dict[stat] = df_dict[stat].rename(columns=rename_feature)
            df_dict[stat] = df_dict[stat].drop(columns=['loannumber'])
        elif stat == 'min':
            rename_feature = {}
            for column in df.columns[2:]:
                rename_feature[column] = column + '_' + stat
            df_dict[stat] = df.groupby(['customerid']).min()
            df_dict[stat] = df_dict[stat].rename(columns=rename_feature)
            df_dict[stat] = df_dict[stat].drop(columns=['loannumber'])
        elif stat == 'mean':
            rename_feature = {}
            for column in df.columns[2:]:
                rename_feature[column] = column + '_' + stat
            df_dict[stat] = df.groupby(['customerid']).mean()
            df_dict[stat] = df_dict[stat].rename(columns=rename_feature)
            df_dict[stat] = df_dict[stat].drop(columns=['loannumber'])

    df_stat = df_dict[stats[0]]

    if len(stats) == 2:
        df_stat = pd.concat([df_dict[stats[0]], df_dict[stats[1]]], axis=1)
    elif len(stats) == 3:
        df_stat = pd.concat([df_dict[stats[0]], df_dict[stats[1]], df_dict[stats[2]]], axis=1)
    else:
        pass

    return df_stat


def transform_demographic(demographics):
    #Process demographic data
    # Drop the features with 80% or greater of missing data
    columns_to_drop = get_percentage_missing_data(demographics, 0.8, show_percentage=False)
    demographics = demographics.drop(columns=columns_to_drop)
    # Nan == employment non defined
    demographics = fill_nan_employment(demographics)
    demographics['birthdate'] = demographics['birthdate'].apply(birthday_to_age)
    rename_mapping = {'birthdate': 'age'}
    demographics = demographics.rename(columns=rename_mapping)

    return demographics


def transform_perf(performance):
    #compute the rate interest and the interest of the loan
    performance = get_interest_data(performance)
    # performance = scale_termdays(performance)
    performance = performance.drop(columns=['approveddate', 'creationdate', 'referredby', 'systemloanid'])
    return performance


def transform_prevloans(prevloans, aggregate_stats=['mean']):
    '''Aggregate all  the previous loans information into one row by client
    Aggregate stats: One between max, mean and min'''
    # Transform prevloans
    # payment info
    prevloans = get_interest_data(prevloans)
    prevloans = scale_termdays(prevloans)
    # date features treatment
    date_features_df = prevloans.drop(columns=get_non_date_features(prevloans))
    # String dates to datetime format
    date_features_df = string_to_datetime(date_features_df)
    # Create a two features tha computes the 'speed' of repayment
    date_features_df = differentiate_dates(date_features_df, ['firstrepaiddate', 'firstduedate'])
    date_features_df = differentiate_dates(date_features_df, ['closeddate', 'creationdate'])
    # Which day of the week the due and the repaid took place
    date_features_df = date_to_day(date_features_df)
    # Merge the transformed date features with the prevloans data
    prevloans = pd.concat(
        [prevloans, date_features_df.drop(columns=get_date_features(prevloans, keep_customer_id=True))], axis=1)
    # Non relevant variables once transformed the data
    columns_to_drop = ['systemloanid', 'approveddate', 'creationdate', 'closeddate', 'referredby', 'firstduedate',
                       'firstrepaiddate']
    prevloans = prevloans.drop(columns=columns_to_drop)
    # final prev loans data
    prevloans = get_statistics_by_customer(prevloans, stats=aggregate_stats)
    prevloans['customerid'] = prevloans.index
    prevloans = prevloans.reset_index(drop=True)

    return prevloans


def merge_data(demographics, prevloans, perf):
    #Merge the 3 dataframes using the customer id
    final_data = perf.merge(demographics, on=['customerid'], how='inner').merge(prevloans, on=['customerid'],
                                                                                how='inner')
    final_data['good_bad_flag'] = final_data['good_bad_flag'].apply(lambda x: 1 if x == 'Good' else 0)
    return final_data


def one_hot_encoding(df):
    '''Search for columns in a string format and one hot encode the categorical values
    customerid and good bad flag are avoided'''
    #encode categorical variables
    onehot_encoding = []
    for column in final_data.columns:
        if isinstance(final_data[column][0], str) and (column != 'good_bad_flag') and (column != 'customerid'):
            onehot_encoding.append(column)
        else:
            pass
    df = pd.get_dummies(df, columns=onehot_encoding)

    return df


if __name__ == '__main__':
    #Data location
    demographic_data_path = 'https://raw.githubusercontent.com/ayoubelqadi/Loan-Default-Prediction/main/data/traindemographics.csv'
    performing_data_path = 'https://raw.githubusercontent.com/ayoubelqadi/Loan-Default-Prediction/main/data/trainperf.csv'
    previous_loans_data_path = 'https://raw.githubusercontent.com/ayoubelqadi/Loan-Default-Prediction/main/data/trainprevloans.csv'
    #Reading data
    train_perf = pd.read_csv(performing_data_path)
    train_prevloans = pd.read_csv(previous_loans_data_path)
    train_demographics = pd.read_csv(demographic_data_path)
    #Processing the data 
    demographics = transform_demographic(train_demographics)
    prevloans = transform_prevloans(train_prevloans, aggregate_stats=['mean'])
    perf = transform_perf(train_perf)
    final_data = merge_data(demographics, prevloans, perf)

    encode_cat_variables = input()
    if encode_cat_variables:
        final_data = one_hot_encoding(final_data)
    else:
        #Drop all categorical features
        cat_features = []
        for feature in final_data.columns:
            if isinstance(final_data[feature][0], str) and (feature != 'good_bad_flag') and (feature != 'customerid'):
                cat_features.append(feature)
            else:
                pass
        final_data = final_data.drop(columns=cat_features)
    columns_to_drop = ['customerid']
    X = final_data.drop(columns=columns_to_drop)

    


