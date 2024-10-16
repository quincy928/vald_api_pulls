import requests
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import time
import os
import urllib.parse
import re
import json
import pytz
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class Vald():
    def __init__(self):
        self.token_url = os.getenv("TOKEN_URL")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.tenant_id = os.getenv("TENANT_ID")

        #self.modified_from_utc = os.getenv("MODIFIED_FROM_UTC")

        self.forcedecks_api_url = 'https://prd-use-api-extforcedecks.valdperformance.com/'
        self.groupnames_api_url = os.getenv("GROUPNAMES_API_URL")
        self.profiles_api_url = os.getenv("PROFILES_API_URL")

        self.vald_master_file_path = os.path.join('data', 'master_files', 'forcedecks_allsports.csv')
        print(self.vald_master_file_path, "valdmaster(FD)")
        self.base_directory = 'data'
    
    def get_last_update(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_row = df.iloc[-1]
        test_date_utc = last_row['adjusted_times']
        last_index = last_row.name
        
        return test_date_utc, last_index
    
    def sanitize_filename(self, filename):
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
        return sanitized

    def sanitize_foldername(self, foldername):
        sanitized = re.sub(r'[^a-zA-Z0-9_ -]', '_', foldername)
        return sanitized
    
    def get_access_token(self):
        auth_response = requests.post(
            self.token_url,
            auth=HTTPBasicAuth(self.client_id, self.client_secret),
            data={'grant_type': 'client_credentials'}
        )
        return auth_response.json()['access_token'] if auth_response.status_code == 200 else None

    def fetch_data(self, url, headers):
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None
    
    def get_tests(self, start_date):
        print(f"Start date: {start_date}")
        
        # Retrieve the access token
        access_token = self.get_access_token()
        if not access_token:
            print("Failed to retrieve access token")
            return pd.DataFrame()  # Return an empty DataFrame if access token retrieval fails
    
        # Set headers using the retrieved access token
        headers = {'Authorization': f'Bearer {access_token}', 'Accept': '*/*'}
    
        # Construct the API URL
        api_url = f"{self.forcedecks_api_url}/tests?TenantId={self.tenant_id}&ModifiedFromUtc={start_date}"
        print(f'The API URL is: {api_url}')
    
        # Fetch tests data
        tests_data = self.fetch_data(api_url, headers)
        if tests_data is None:
            print("No test data returned.")
            return pd.DataFrame(), start_date  # Return an empty DataFrame if no tests data is returned
        
        print(f"Number of tests retrieved: {len(tests_data['tests'])}")
    
        # Fetch group names using the /groupnames endpoint
        api_url_groupnames = f"{self.groupnames_api_url}?TenantId={self.tenant_id}"
        group_data = self.fetch_data(api_url_groupnames, headers)
        id_to_name = {group['id']: group['name'] for group in group_data['groups']}
        #print(f"Group names mapping: {id_to_name}")
    
        # Prepare to collect additional trial metrics for each test
        test_ids = [test['testId'] for test in tests_data['tests']]
        metrics_list = []

        
        
        # Fetch athlete profile information first
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Fetch profiles concurrently
            profile_futures = {
                test['testId']: executor.submit(
                    self.fetch_data, 
                    f"{self.profiles_api_url}{test['profileId']}?TenantId={self.tenant_id}", 
                    headers
                )
                for test in tests_data['tests']
            }
        
            # Process the athlete profile information
            for test in tests_data['tests']:
                test_id = test['testId']
                print(f"Processing profile for testId: {test_id}")
        
                try:
                    # Retrieve athlete profile
                    profile_result = profile_futures[test_id].result()
                    #print(f'Profile_result is {profile_result}')
                    if profile_result is not None:
                        test['Name'] = f"{profile_result['givenName'].strip()} {profile_result['familyName'].strip()}"
                        group_ids = profile_result['groupIds']
                        group_names = [id_to_name.get(g_id, "ID not found") for g_id in group_ids]
                        test['Groups'] = '|'.join(group_names)
                        print(f"Profile for testId {test_id}: Name={test['Name']}, Groups={test['Groups']}")
                    else:
                        # Make a separate call to the API
                        retry_profile_url = f"{self.profiles_api_url}{test['profileId']}?TenantId={self.tenant_id}"
                        retry_profile_result = self.fetch_data(retry_profile_url, headers)
                        
                        if retry_profile_result is not None:
                            test['Name'] = f"{retry_profile_result['givenName'].strip()} {retry_profile_result['familyName'].strip()}"
                            group_ids = retry_profile_result['groupIds']
                            group_names = [id_to_name.get(g_id, "ID not found") for g_id in group_ids]
                            test['Groups'] = '|'.join(group_names)
                            print(f"Profile (after retry) for testId {test_id}: Name={test['Name']}, Groups={test['Groups']}")
                        else:
                            print(f"Retry failed: No profile data for testId {test_id}, athleteId = {test['profileId']}")
        
                except Exception as e:
                    print(f"An error occurred while processing profile for testId {test_id}: {e}")
                    continue  # Skip to the next test if there's an error
        
        # Now fetch trial metrics
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Fetch trials concurrently
            trial_futures = {
                test['testId']: executor.submit(
                    self.fetch_data, 
                    f"https://prd-use-api-extforcedecks.valdperformance.com/v2019q3/teams/{self.tenant_id}/tests/{test['testId']}/trials", 
                    headers
                )
                for test in tests_data['tests']
            }
        
            # Process the trial metrics
            for test in tests_data['tests']:
                test_id = test['testId']
                print(f"Processing trials for testId: {test_id}")
        
                try:
                    # Retrieve trial metrics
                    trial_result = trial_futures[test_id].result()                  
                    
                    if trial_result is not None:
                        print(f"Processing trials for testId {test_id}, number of reps: {len(trial_result)}")
                        l_trial_result = [trial for trial in trial_result if trial['limb'] == 'Left']
                        r_trial_result = [trial for trial in trial_result if trial['limb'] == 'Right']
                        test['Reps'] = len(trial_result)
                        test['Reps (L)'] = len(l_trial_result)
                        test['Reps (R)'] = len(r_trial_result)
                        records = []
                        max_trial_data = {}
        
                        for record in trial_result:
                            # Handle each result within the record
                            for result in record['results']:
                                result_name = result['definition']['name']  # Get the result name
                                trial_type = result['limb']  # Get the trial type (Left, Right, Trial, Asym, etc.)
                                result_value = result['value']  # Get the corresponding value
        
                                if ((test['testType'].startswith('SL')) | (test['testType'].startswith('RSKIP'))):
                                    # In single leg tests, limb will always be 'Trial'
                                    limb = record['limb']
                                    if limb in ['Left', 'Right']:  # Check if the limb is Left or Right
                                        # Create a unique column name for single leg tests
                                        column_name = f"{result_name}_{limb}"
        
                                        # If the column already exists, update it with the max value across reps
                                        if column_name in max_trial_data:
                                            max_trial_data[column_name] = max(max_trial_data[column_name], result_value)
                                        else:
                                            max_trial_data[column_name] = result_value
        
                                else:
                                    # Handle regular tests
                                    # Compose a unique column name combining result name and trial type
                                    column_name = f"{result_name}_{trial_type}"
        
                                    # If the column already exists, update it with the max value across reps
                                    if column_name in max_trial_data:
                                        if trial_type == 'Asym':
                                            max_trial_data[column_name] = result_value if abs(result_value) > abs(max_trial_data[column_name]) else max_trial_data[column_name]
                                        else:
                                            max_trial_data[column_name] = max(max_trial_data[column_name], result_value)
                                    else:
                                        max_trial_data[column_name] = result_value
        
                        # After processing all records, append the max values to records
                        records.append(max_trial_data)
        
                        # Step 3: Create a DataFrame from the records list for the current test
                        metrics_df = pd.DataFrame(records)
                        metrics_df['testId'] = test_id  # Add testId to align with the test data
                        metrics_list.append(metrics_df)  # Append the DataFrame to the list
        
                    else:
                        print(f"No trial data for testId {test_id}, trying again...")
                        # Make a separate call to the API for retry
                        retry_trial_url = f"https://prd-use-api-extforcedecks.valdperformance.com/v2019q3/teams/{self.tenant_id}/tests/{test_id}/trials"
                        retry_trial_result = self.fetch_data(retry_trial_url, headers)
                        if retry_trial_result is not None:
                            print(f"Processing trials for testId {test_id} (after retry), number of reps: {len(retry_trial_result)}")
                            l_trial_result = [trial for trial in retry_trial_result if trial['limb'] == 'Left']
                            r_trial_result = [trial for trial in retry_trial_result if trial['limb'] == 'Right']
                            test['Reps'] = len(retry_trial_result)
                            test['Reps (L)'] = len(l_trial_result)
                            test['Reps (R)'] = len(r_trial_result)
                            records = []
                            max_trial_data = {}
                            
                            # Extract the relevant data from each record
                            for record in retry_trial_result:
                                # Handle each result within the record
                                for result in record['results']:
                                    result_name = result['definition']['name']  # Get the result name
                                    trial_type = result['limb']  # Get the trial type (Left, Right, Trial, Asym, etc.)
                                    result_value = result['value']  # Get the corresponding value
        
                                    if ((test['testType'].startswith('SL')) | (test['testType'].startswith('RSKIP'))):
                                        # In single leg tests, limb will always be 'Trial'
                                        limb = result['limb']
                                        if limb in ['Left', 'Right']:  # Check if the limb is Left or Right
                                            # Create a unique column name for single leg tests
                                            column_name = f"{result_name}_{limb}"
        
                                            # If the column already exists, update it with the max value across reps
                                            if column_name in max_trial_data:
                                                max_trial_data[column_name] = max(max_trial_data[column_name], result_value)
                                            else:
                                                max_trial_data[column_name] = result_value
        
                                    else:
                                        # Handle regular tests
                                        # Compose a unique column name combining result name and trial type
                                        column_name = f"{result_name}_{trial_type}"
        
                                        # If the column already exists, update it with the max value across reps
                                        if column_name in max_trial_data:
                                            if trial_type == 'Asym':
                                                max_trial_data[column_name] = result_value if abs(result_value) > abs(max_trial_data[column_name]) else max_trial_data[column_name]
                                            else:
                                                max_trial_data[column_name] = max(max_trial_data[column_name], result_value)
                                        else:
                                            max_trial_data[column_name] = result_value
        
                            # After processing all records, append the max values to records
                            records.append(max_trial_data)
        
                            # Step 3: Create a DataFrame from the records list for the current test
                            metrics_df = pd.DataFrame(records)
                            metrics_df['testId'] = test_id  # Add testId to align with the test data
                            metrics_list.append(metrics_df)  # Append the DataFrame to the list
                        else:
                            print(f"Retry failed: No trial data for testId {test_id}")
        
                except Exception as e:
                    print(f"An error occurred while processing trials for testId {test_id}: {e}")
                    continue  # Skip to the next test if there's an error
        
        print("Data retrieval complete.")


    
        # Convert tests_data['tests'] to DataFrame
        tests_df = pd.DataFrame(tests_data['tests'])
        tests_df['Tag'] = tests_df['attributes'].apply(lambda x: x[0]['attributeValueName'] if isinstance(x, list) and len(x) > 0 else None)
        tests_df['Tag Type'] = tests_df['attributes'].apply(lambda x: x[0]['attributeTypeName'] if isinstance(x, list) and len(x) > 0 else None)

        if (tests_df['testType'] == 'SLSB').any():
            tests_df['Exercise Length [S]'] = tests_df['parameter'].apply(
                lambda x: x['value'] if isinstance(x, dict) and x.get('resultId') == 655381 else None
            )
            
            tests_df['Eyes Closed'] = tests_df['extendedParameters'].apply(
                lambda x: next((param['value'] for param in x if isinstance(param, dict) and param.get('resultId') == 655382), None)
            )
            
            tests_df['Unstable Surface'] = tests_df['extendedParameters'].apply(
                lambda x: next((param['value'] for param in x if isinstance(param, dict) and param.get('resultId') == 655383), None)
            )
            
            tests_df['Secondary Task'] = tests_df['extendedParameters'].apply(
                lambda x: next((param['value'] for param in x if isinstance(param, dict) and param.get('resultId') == 655384), None)
            )
        
        # Clean up the DataFrame if needed, e.g., drop the original 'parameter' and 'extendedParameters' columns
        tests_df = tests_df.drop(columns=['parameter', 'extendedParameters'], errors='ignore')
        pst = pytz.timezone('America/Los_Angeles')
        tests_df['adjusted_times'] = tests_df['modifiedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))

    
        # Concatenate all the metrics DataFrames and merge with tests_df based on 'testId'
        if metrics_list:
            metrics_df = pd.concat(metrics_list, axis=0)
            #print("Metrics DataFrame:")

        else:
            print("No metrics data found.")
            metrics_df = pd.DataFrame()  # Return empty DataFrame if no metrics are found
    
        # Ensure both DataFrames have the 'testId' column and merge them
        merged_df = pd.merge(tests_df, metrics_df, on='testId', how='left')
        #print("Merged DataFrame:")

    
        # Replace the original tests_data['tests'] with the merged DataFrame
        tests_data['tests'] = merged_df.to_dict(orient='records')
        if 'tests' in tests_data and tests_data['tests']:
            last_date = max(test['adjusted_times'] for test in tests_data['tests'] if 'adjusted_times' in test)
            print(f"The latest modifiedDateUtc in this match is: {last_date}")
        else:
            print("No tests data available.")
        return pd.json_normalize(tests_data['tests']), last_date
    
    def format_CMJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns={'testType': 'Test Type',
                                'weight': 'BW [KG]',
                                'Jump Height (Imp-Mom)_Trial': 'Jump Height (Imp-Mom) [cm]',
                                'Peak Power_Trial': 'Peak Power [W]',
                                'Peak Power / BM_Trial': 'Peak Power / BM [W/kg]',
                                'Concentric Impulse (Abs) / BM_Trial': 'Concentric Impulse (Abs) / BM [N s]',
                                'Concentric Impulse-100ms_Trial': 'Concentric Impulse-100ms [N s]',
                                'Eccentric Deceleration Impulse / BM_Trial': 'Eccentric Deceleration Impulse / BM [N s/kg]',
                                'Concentric RFD / BM_Trial': 'Concentric RFD / BM [N/s/kg]',
                                'Eccentric Deceleration RFD / BM_Trial': 'Eccentric Deceleration RFD / BM [N/s/kg]',
                                'Concentric Peak Force / BM_Trial': 'Concentric Peak Force / BM [N/kg]',
                                'Eccentric Peak Force / BM_Trial': 'Eccentric Peak Force / BM [N/kg]',
                                'RSI-modified (Imp-Mom)_Trial': 'RSI-modified (Imp-Mom) [m/s]',
                                'Contraction Time_Trial': 'Contraction Time [ms]',
                                'Countermovement Depth_Trial': 'Countermovement Depth [cm]',
                                'Force at Zero Velocity_Trial': 'Force at Zero Velocity [N]',
                                'Force at Peak Power_Trial': 'Force at Peak Power [N]',
                                'Velocity at Peak Power_Trial': 'Velocity at Peak Power [m/s]',
                                'Concentric Impulse_Asym': 'Concentric Impulse % (Asym) (%)',
                                'Eccentric Deceleration Impulse_Asym': 'Eccentric Deceleration Impulse % (Asym) (%)',
                                'Landing Impulse_Asym': 'Landing Impulse % (Asym) (%)',
                                'Eccentric Deceleration Impulse_Trial': 'Eccentric Deceleration Impulse [N s]',
                                'Concentric Impulse (Abs) / BM_Trial': 'Concentric Impulse (Abs) / BM [N s]',
                                'Eccentric Braking Impulse_Trial': 'Eccentric Braking Impulse [N s]',
                                'Takeoff Peak Force / BM_Trial': 'Takeoff Peak Force / BM [N/kg]',
                                'Vertical Velocity at Takeoff_Trial': 'Vertical Velocity at Takeoff [m/s]'
                            })
        columns_to_keep = [
                            'Name',
                            'Test Type',
                            'Date',
                            'Time',
                            'BW [KG]',
                            'Reps',
                            'Jump Height (Imp-Mom) [cm]', 
                            'Peak Power [W]', 
                            'Peak Power / BM [W/kg]', 
                            'Concentric Impulse-100ms [N s]', 
                            'Concentric Impulse (Abs) / BM [N s]', 
                            'Eccentric Deceleration Impulse / BM [N s/kg]', 
                            'Concentric RFD / BM [N/s/kg]', 
                            'Eccentric Deceleration RFD / BM [N/s/kg]', 
                            'Concentric Peak Force / BM [N/kg]', 
                            'Eccentric Peak Force / BM [N/kg]', 
                            'RSI-modified (Imp-Mom) [m/s]', 
                            'Contraction Time [ms]', 
                            'Countermovement Depth [cm]', 
                            'Force at Zero Velocity [N]', 
                            'Force at Peak Power [N]', 
                            'Velocity at Peak Power [m/s]', 
                            'Concentric Impulse % (Asym) (%)',
                            'Eccentric Deceleration Impulse % (Asym) (%)',
                            'Landing Impulse % (Asym) (%)',
                            'Eccentric Deceleration Impulse [N s]',
                            'Concentric Impulse (Abs) / BM [N s]',
                            'Eccentric Braking Impulse [N s]',
                            'Takeoff Peak Force / BM [N/kg]',
                            'Tag',
                            'Tag Type',
                            'Vertical Velocity at Takeoff [m/s]'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df['RSI-modified (Imp-Mom) [m/s]'] = df['RSI-modified (Imp-Mom) [m/s]']*0.01
        df['Contraction Time [ms]'] = df['Contraction Time [ms]']*1000
        df = df.reindex(columns=columns_to_keep).round(2)

        return df

    def format_ISOSQT(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns={'Name': 'Name',
                                'testType': 'Test Type',
                                'weight': 'BW [KG]',
                                'Peak Vertical Force_Trial': 'Peak Vertical Force [N]',
                                'Peak Vertical Force / BM_Trial': 'Peak Vertical Force / BM [N/kg]',
                                'Peak Vertical Force_Left': 'Peak Vertical Force [N] (L)',
                                'Peak Vertical Force_Right': 'Peak Vertical Force [N] (R)',
                                'Peak Vertical Force_Asym': 'Peak Vertical Force % (Asym) (%)',
                                'Start Time to Peak Force_Trial': 'Start Time to Peak Force [s]',
                                'RFD - 100ms_Trial': 'RFD - 100ms [N/s]',
                                'RFD - 200ms_Trial': 'RFD - 200ms [N/s]',
                                'Net Force at 100ms_Trial': 'Net Force at 100ms [N]',
                                'Net Force at 200ms_Trial': 'Net Force at 200ms [N]'
                            })
        columns_to_keep = ['Name',
                   'Test Type',
                   'Date',
                   'Time',
                   'BW [KG]',
                   'Reps',
                   'Peak Vertical Force [N]', 
                   'Peak Vertical Force / BM [N/kg]', 
                   'Peak Vertical Force [N] (L)',
                   'Peak Vertical Force [N] (R)',
                   'Peak Vertical Force % (Asym) (%)',
                   'Start Time to Peak Force [s]', 
                   'RFD - 100ms [N/s]', 
                   'RFD - 200ms [N/s]', 
                   'Net Force at 100ms [N]', 
                   'Net Force at 200ms [N]',
                   'Tag',
                   'Tag Type']
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_SJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns={'Name': 'Name',
                                'testType': 'Test Type',
                                'weight': 'BW [KG]',
                                'Jump Height (Imp-Mom)_Trial': 'Jump Height (Imp-Mom) [cm]',
                                'Peak Landing Force_Trial': 'Peak Landing Force [N]',
                                'Peak Landing Force / BM_Trial': 'Peak Landing Force / BM [N/kg]',
                                'Peak Power / BM_Trial': 'Peak Power / BM [W/kg]',
                                'RSI-modified (Imp-Mom)_Trial': 'RSI-modified (Imp-Mom) [m/s]',
                                'Start Time to Peak Force_Trial': 'Start Time to Peak Force [s]'
                            })
        columns_to_keep = ['Name',
                   'Test Type',
                   'Date',
                   'Time',
                   'BW [KG]',
                   'Reps',
                   'Additional Load [kg]',
                   'Jump Height (Imp-Mom) [cm]', 
                   'Peak Landing Force [N]', 
                   'Peak Landing Force / BM [N/kg]', 
                   'Peak Power / BM [W/kg]', 
                   'RSI-modified (Imp-Mom) [m/s]',
                   'Tag',
                   'Tag Type']
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_SLJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns={'Name': 'Name',
                                'testType': 'Test Type',
                                'Date': 'Date',
                                'Time': 'Time',
                                'weight': 'BW [KG]',
                                'Reps_Left': 'Reps (L)',
                                'Reps_Right': 'Reps (R)',
                                'Jump Height (Imp-Mom)_Left': 'Jump Height (Imp-Mom) [cm] (L)', 
                                'Jump Height (Imp-Mom)_Right': 'Jump Height (Imp-Mom) [cm] (R)', 
                                'Peak Landing Force_Left': 'Peak Landing Force [N] (L)', 
                                'Peak Landing Force_Right': 'Peak Landing Force [N] (R)', 
                                'Peak Power / BM_Left': 'Peak Power / BM [W/kg] (L)', 
                                'Peak Power / BM_Right': 'Peak Power / BM [W/kg] (R)', 
                                'RSI-modified_Left': 'RSI-modified [m/s] (L)', 
                                'RSI-modified_Right': 'RSI-modified [m/s] (R)'
                            })
        columns_to_keep = ['Name',
                            'Test Type',
                            'Date',
                            'Time',
                            'BW [KG]',
                            'Reps (L)',
                            'Reps (R)',
                            'Jump Height (Imp-Mom) [cm]',
                            'Jump Height (Imp-Mom) [cm] (L)',
                            'Jump Height (Imp-Mom) [cm] (R)',
                            'Jump Height (Imp-Mom) [cm] (Asym)(%)',
                            'Peak Landing Force [N]',
                            'Peak Landing Force [N] (L)',
                            'Peak Landing Force [N] (R)',
                            'Peak Landing Force [N] (Asym)(%)',
                            'Peak Power / BM [W/kg]',
                            'Peak Power / BM [W/kg] (L)',
                            'Peak Power / BM [W/kg] (R)',
                            'Peak Power / BM [W/kg] (Asym)(%)',
                            'RSI-modified [m/s]',
                            'RSI-modified [m/s] (L)',
                            'RSI-modified [m/s] (R)',
                            'RSI-modified [m/s] (Asym)(%)',
                            'Tag',
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        
        df['Jump Height (Imp-Mom) [cm]'] = np.maximum(df['Jump Height (Imp-Mom) [cm] (L)'], df['Jump Height (Imp-Mom) [cm] (R)'])
        df['Peak Landing Force [N]'] = np.maximum(df['Peak Landing Force [N] (L)'], df['Peak Landing Force [N] (R)'])
        df['Peak Power / BM [W/kg]'] = np.maximum(df['Peak Power / BM [W/kg] (L)'], df['Peak Power / BM [W/kg] (R)'])
        df['RSI-modified [m/s]'] = np.maximum(df['RSI-modified [m/s] (L)'], df['RSI-modified [m/s] (R)'])        

        df['Jump Height (Imp-Mom) [cm] (Asym)(%)'] = (np.minimum(df['Jump Height (Imp-Mom) [cm] (L)'], df['Jump Height (Imp-Mom) [cm] (R)']) / df['Jump Height (Imp-Mom) [cm]'] - 1) * 100
        df['Jump Height (Imp-Mom) [cm] (Asym)(%)'] = np.where(df['Jump Height (Imp-Mom) [cm] (R)'] > df['Jump Height (Imp-Mom) [cm] (L)'],
                                                               abs(df['Jump Height (Imp-Mom) [cm] (Asym)(%)']),
                                                               df['Jump Height (Imp-Mom) [cm] (Asym)(%)'])
        

        df['Peak Landing Force [N] (Asym)(%)'] = (np.minimum(df['Peak Landing Force [N] (L)'], df['Peak Landing Force [N] (R)']) / df['Peak Landing Force [N]'] - 1) * 100
        df['Peak Landing Force [N] (Asym)(%)'] = np.where(df['Peak Landing Force [N] (R)'] > df['Peak Landing Force [N] (L)'],
                                                          abs(df['Peak Landing Force [N] (Asym)(%)']),
                                                          df['Peak Landing Force [N] (Asym)(%)'])
        

        df['Peak Power / BM [W/kg] (Asym)(%)'] = (np.minimum(df['Peak Power / BM [W/kg] (L)'], df['Peak Power / BM [W/kg] (R)']) / df['Peak Power / BM [W/kg]'] - 1) * 100
        df['Peak Power / BM [W/kg] (Asym)(%)'] = np.where(df['Peak Power / BM [W/kg] (R)'] > df['Peak Power / BM [W/kg] (L)'],
                                                          abs(df['Peak Power / BM [W/kg] (Asym)(%)']),
                                                          df['Peak Power / BM [W/kg] (Asym)(%)'])
        

        df['RSI-modified [m/s] (Asym)(%)'] = (np.minimum(df['RSI-modified [m/s] (L)'], df['RSI-modified [m/s] (R)']) / df['RSI-modified [m/s]'] - 1) * 100
        df['RSI-modified [m/s] (Asym)(%)'] = np.where(df['RSI-modified [m/s] (R)'] > df['RSI-modified [m/s] (L)'],
                                                      abs(df['RSI-modified [m/s] (Asym)(%)']),
                                                      df['RSI-modified [m/s] (Asym)(%)'])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_IMTP(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns={
                        "testType": "Test Type",
                        "weight": "BW [KG]",
                        "Peak Vertical Force_Trial": "Peak Vertical Force [N]",
                        "Peak Vertical Force / BM_Trial": "Peak Vertical Force / BM [N/kg]",
                        "Peak Vertical Force_Left": "Peak Vertical Force [N] (L)",
                        "Peak Vertical Force_Right": "Peak Vertical Force [N] (R)",
                        "Peak Vertical Force_Asym": "Peak Vertical Force % (Asym) (%)",
                        "Start Time to Peak Force_Trial": "Start Time to Peak Force [s]",
                        "RFD - 100ms_Trial": "RFD - 100ms [N/s]",
                        "RFD - 200ms_Trial": "RFD - 200ms [N/s]",
                        "Net Force at 100ms_Trial": "Net Force at 100ms [N]",
                        "Net Force at 200ms_Trial": "Net Force at 200ms [N]",
                    })
        columns_to_keep = ["Name",
            "Test Type",
            "Date",
            "Time",
            "BW [KG]",
            "Reps",
            "Peak Vertical Force [N]",
            "Peak Vertical Force / BM [N/kg]",
            "Peak Vertical Force [N] (L)",
            "Peak Vertical Force [N] (R)",
            "Peak Vertical Force % (Asym) (%)",
            "Start Time to Peak Force [s]",
            "RFD - 100ms [N/s]",
            "RFD - 200ms [N/s]",
            "Net Force at 100ms [N]",
            "Net Force at 200ms [N]",
            "Tag",
            'Tag type'
        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        
        
        # Define the columns you want to keep (after renaming)
        
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_DJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {"Athlete Standing Weight_Trial": "Athlete Standing Weight [kg]",
                                  "weight": 'BW [KG]',
                                  'testType': 'Test Type',
                                    "Contact Time_Trial": "Contact Time [s]",
                                    "Countermovement Depth_Trial": "Countermovement Depth [cm]",
                                    "Eccentric Duration_Trial": "Eccentric Duration [ms]",
                                    "Force at Zero Velocity / BM_Trial": "Force at Zero Velocity / BM [N/kg]",
                                    "Jump Height (Imp-Mom)_Trial": "Jump Height (Imp-Mom) [cm]",
                                    "RSI (JH (Flight Time)/Contact Time)_Trial": "RSI (JH (Flight Time)/Contact Time) [m/s]",
                                    "Active Stiffness_Asym": "Active Stiffness % (Asym) (%)",
                                    "Eccentric:Concentric Mean Force Ratio_Asym": "Eccentric:Concentric Mean Force Ratio % (Asym) (%)",
                                    "Eccentric Impulse_Asym": "Eccentric Impulse % (Asym) (%)",
                                    "Eccentric Mean Force_Asym": "Eccentric Mean Force % (Asym) (%)",
                                })
        
        # Define the columns you want to keep (after renaming)
        columns_to_keep = [
                        "Name",
                        "Test Type",
                        "Date",
                        "Time",
                        "BW [KG]",
                        "Reps",
                        "Athlete Standing Weight [kg]",
                        "Contact Time [s]",
                        "Countermovement Depth [cm]",
                        "Eccentric Duration [ms]",
                        "Force at Zero Velocity / BM [N/kg]",
                        "Jump Height (Imp-Mom) [cm]",
                        "RSI (JH (Flight Time)/Contact Time) [m/s]",
                        "Active Stiffness % (Asym) (%)",
                        "Eccentric:Concentric Mean Force Ratio % (Asym) (%)",
                        "Eccentric Impulse % (Asym) (%)",
                        "Eccentric Mean Force % (Asym) (%)",
                        "Tag",
                        'Tag Type'
                    ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df['Eccentric Duration [ms]'] = df['Eccentric Duration [ms]'] * 1000
        df["RSI (JH (Flight Time)/Contact Time) [m/s]"] = df["RSI (JH (Flight Time)/Contact Time) [m/s]"] * 0.01
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_LAH(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {
                                    "weight": "BW [KG]",
                                    "testType": "Test Type",
                                    "Drop Landing_Trial": "Drop Landing [s]",
                                    "Peak Drop Landing Force_Trial": "Peak Drop Landing Force [N]",
                                    "Time to Stabilisation_Trial": "Time to Stabilisation [s]",
                                    "Peak Drop Landing Force_Asym": "Peak Drop Landing Force % (Asym) (%)",
                                    "Time to Stabilisation_Asym": "Time to Stabilisation % (Asym) (%)",
                                })
        columns_to_keep = [
                            "Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps",
                            "Drop Landing [s]",
                            "Peak Drop Landing Force [N]",
                            "Time to Stabilisation [s]",
                            "Peak Drop Landing Force % (Asym) (%)",
                            "Time to Stabilisation % (Asym) (%)",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_SLLAH(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {"Time to Stabilisation_Left": "Time to Stabilisation [s] (L)",
                                   "weight": "BW [KG]",
                                  "testType": "Test Type",
                                    "Time to Stabilisation_Right": "Time to Stabilisation [s] (R)",
                                    "Time to Stabilisation_Asym": "Time to Stabilisation [s] (Asym)(%)",
                                    "Peak Drop Landing Force_Left": "Peak Drop Landing Force [N] (L)",
                                    "Peak Drop Landing Force_Right": "Peak Drop Landing Force [N] (R)",
                                    "Peak Drop Landing Force_Asym": "Peak Drop Landing Force [N] (Asym)(%)",
                                })
        columns_to_keep = [
                            "Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps (L)",
                            "Reps (R)",
                            "Time to Stabilisation [s]",
                            "Time to Stabilisation [s] (L)",
                            "Time to Stabilisation [s] (R)",
                            "Time to Stabilisation [s] (Asym)(%)",
                            "Peak Drop Landing Force [N]",
                            "Peak Drop Landing Force [N] (L)",
                            "Peak Drop Landing Force [N] (R)",
                            "Peak Drop Landing Force [N] (Asym)(%)",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df['Time to Stabilisation [s]'] = np.maximum(df['Time to Stabilisation [s] (L)'], df['Time to Stabilisation [s] (R)'])
        df["Peak Drop Landing Force [N]"] = np.maximum(df['Peak Drop Landing Force [N] (L)'], df['Peak Drop Landing Force [N] (R)'])

        df['Time to Stabilisation [s] (Asym)(%)'] = (np.minimum(df['Time to Stabilisation [s] (L)'], df['Time to Stabilisation [s] (R)']) / df['Time to Stabilisation [s]'] - 1) * 100
        df['Time to Stabilisation [s] (Asym)(%)'] = np.where(df['Time to Stabilisation [s] (R)'] > df['Time to Stabilisation [s] (L)'],
                                                               abs(df['Time to Stabilisation [s] (Asym)(%)']),
                                                               df['Time to Stabilisation [s] (Asym)(%)'])
        

        df['Peak Drop Landing Force [N] (Asym)(%)'] = (np.minimum(df['Peak Drop Landing Force [N] (L)'], df['Peak Drop Landing Force [N] (R)']) / df['Peak Drop Landing Force [N]'] - 1) * 100
        df['Peak Drop Landing Force [N] (Asym)(%)'] = np.where(df['Peak Drop Landing Force [N] (R)'] > df['Peak Drop Landing Force [N] (L)'],
                                                          abs(df['Peak Drop Landing Force [N] (Asym)(%)']),
                                                          df['Peak Drop Landing Force [N] (Asym)(%)'])
        # Define the columns you want to keep (after renaming)
        
        df = df.reindex(columns=columns_to_keep).round(2)
        return df


    def format_HJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {
                        "weight": "BW [KG]",
                        "testType": "Test Type",
                        "Active Stiffness_Trial": "Active Stiffness [N/m]",
                        "Number of Hops/Reps_Trial": "Number of Hops/Reps",
                        "Fatigue Hops/Reps_Trial": "Fatigue Hops/Reps",
                        "Contact Time_Trial": "Contact Time [ms]",
                        "RSI (Flight/Contact Time)_Trial": "RSI (Flight/Contact Time)",
                        "RSI (Jump Height/Contact Time)_Trial": "RSI (Jump Height/Contact Time) [m/s]",
                        "Landing RFD_Trial": "Landing RFD [N/s]",
                        "Mean Active Stiffness_Trial": "Mean Active Stiffness [N/m]",
                        "Mean Contact Time_Trial": "Mean Contact Time [ms]",
                        "Mean RSI (Flight/Contact Time)_Trial": "Mean RSI (Flight/Contact Time)",
                        "Mean RSI (Jump Height/Contact Time)_Trial": "Mean RSI (Jump Height/Contact Time) [m/s]",
                        "Average Power Fatigue_Trial": "Average Power Fatigue [W]",
                        "Fatigue Hops_Reps_Trial": "Fatigue Hops/Reps",
                        "Stiffness Fatigue_Trial": "Stiffness Fatigue [%]",
                        "Mean Jump Height (Flight Time)_Trial": "Mean Jump Height (Flight Time) [cm]",
                        "Peak Power_Trial": "Peak Power [W]",
                        "Peak Force_Trial": "Peak Force [N]",
                        "Impulse_Asym": "Impulse % (Asym) (%)",
                        "Best Peak Power_Trial": "Best Peak Power [W]",
                        "Best Peak Force_Trial": "Best Peak Force [N]",
                        "Best RSI (Jump Height/Contact Time)_Trial": "Best RSI (Jump Height/Contact Time) [m/s]",
                    })

        columns_to_keep = [
                            "Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "Reps",
                            "BW [KG]",
                            "Active Stiffness [N/m]",
                            "Number of Hops/Reps",
                            "Contact Time [ms]",
                            "RSI (Flight/Contact Time)",
                            "RSI (Jump Height/Contact Time) [m/s]",
                            "Landing RFD [N/s]",
                            "Mean Active Stiffness [N/m]",
                            "Mean Contact Time [ms]",
                            "Mean RSI (Flight/Contact Time)",
                            "Mean RSI (Jump Height/Contact Time) [m/s]",
                            "Average Power Fatigue [W]",
                            "Fatigue Hops/Reps",
                            "Stiffness Fatigue [%]",
                            "Mean Jump Height (Flight Time) [cm]",
                            "Peak Power [W]",
                            "Peak Force [N]",
                            "Impulse % (Asym) (%)",
                            "Best Peak Power [W]",
                            "Best Peak Force [N]",
                            "Best RSI (Jump Height/Contact Time) [m/s]",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        df['Contact Time [ms]'] = df['Contact Time [ms]'] * 100
        df['RSI (Jump Height/Contact Time) [m/s]'] = df['RSI (Jump Height/Contact Time) [m/s]'] * 0.01
        df['Mean Contact Time [ms]'] = df['Mean Contact Time [ms]'] * 1000
        df['Mean RSI (Jump Height/Contact Time) [m/s]'] = df['Mean RSI (Jump Height/Contact Time) [m/s]'] * 0.01
        df['Best RSI (Jump Height/Contact Time) [m/s]'] = df['Best RSI (Jump Height/Contact Time) [m/s]'] * 0.01
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_SLHJ(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {
                        "weight": "BW [KG]",
                        "testType": "Test Type",
                        "Number of Hops/Reps_Left": "Number of Hops/Reps (L)",
                        "Number of Hops/Reps_Right": "Number of Hops/Reps (R)",
                        "Fatigue Hops/Reps_Left": "Fatigue Hops/Reps (L)",
                        "Fatigue Hops/Reps_Right": "Fatigue Hops/Reps (R)",
                        "Contact Time_Left": "Contact Time [ms] (L)",
                        "Contact Time_Right": "Contact Time [ms] (R)",
                        "Impulse_Left": "Impulse [N s] (L)",
                        "Impulse_Right": "Impulse [N s] (R)",
                        "RSI (Flight/Contact Time)_Left": "RSI (Flight/Contact Time) (L)",
                        "RSI (Flight/Contact Time)_Right": "RSI (Flight/Contact Time) (R)",
                        "Mean Active Stiffness_Left": "Mean Active Stiffness [N/m] (L)",
                        "Mean Active Stiffness_Right": "Mean Active Stiffness [N/m] (R)",
                        "Mean Average Power_Left": "Mean Average Power [W] (L)",
                        "Mean Average Power_Right": "Mean Average Power [W] (R)",
                        "Average Power Fatigue_Left": "Average Power Fatigue [W] (L)",
                        "Average Power Fatigue_Right": "Average Power Fatigue [W] (R)",
                        "Landing RFD Fatigue_Left": "Landing RFD Fatigue [%] (L)",
                        "Landing RFD Fatigue_Right": "Landing RFD Fatigue [%] (R)",
                        "RSI (Flight/Contact Time) Fatigue_Left": "RSI (Flight/Contact Time) Fatigue [%] (L)",
                        "RSI (Flight/Contact Time) Fatigue_Right": "RSI (Flight/Contact Time) Fatigue [%] (R)",
                        "Stiffness Fatigue_Left": "Stiffness Fatigue [%] (L)",
                        "Stiffness Fatigue_Right": "Stiffness Fatigue [%] (R)"
                    })
        columns_to_keep =["Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps (L)",
                            "Reps (R)",
                            "Number of Hops/Reps",
                            "Number of Hops/Reps (L)",
                            "Number of Hops/Reps (R)",
                            "Number of Hops/Reps (Asym)(%)",
                            "Contact Time [ms]",
                            "Contact Time [ms] (L)",
                            "Contact Time [ms] (R)",
                            "Contact Time [ms] (Asym)(%)",
                            "Impulse [N s]",
                            "Impulse [N s] (L)",
                            "Impulse [N s] (R)",
                            "Impulse [N s] (Asym)(%)",
                            "RSI (Flight/Contact Time)",
                            "RSI (Flight/Contact Time) (L)",
                            "RSI (Flight/Contact Time) (R)",
                            "RSI (Flight/Contact Time) (Asym)(%)",
                            "Mean Active Stiffness [N/m]",
                            "Mean Active Stiffness [N/m] (L)",
                            "Mean Active Stiffness [N/m] (R)",
                            "Mean Active Stiffness [N/m] (Asym)(%)",
                            "Mean Average Power [W]",
                            "Mean Average Power [W] (L)",
                            "Mean Average Power [W] (R)",
                            "Mean Average Power [W] (Asym)(%)",
                            "Average Power Fatigue [W]",
                            "Average Power Fatigue [W] (L)",
                            "Average Power Fatigue [W] (R)",
                            "Average Power Fatigue [W] (Asym)(%)",
                            "Landing RFD Fatigue [%]",
                            "Landing RFD Fatigue [%] (L)",
                            "Landing RFD Fatigue [%] (R)",
                            "Landing RFD Fatigue [%] (Asym)(%)",
                            "RSI (Flight/Contact Time) Fatigue [%]",
                            "RSI (Flight/Contact Time) Fatigue [%] (L)",
                            "RSI (Flight/Contact Time) Fatigue [%] (R)",
                            "RSI (Flight/Contact Time) Fatigue [%] (Asym)(%)",
                            "Stiffness Fatigue [%]",
                            "Stiffness Fatigue [%] (L)",
                            "Stiffness Fatigue [%] (R)",
                            "Stiffness Fatigue [%] (Asym)(%)",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0

        df['Contact Time [ms] (L)'] = df['Contact Time [ms] (L)'] * 1000
        df['Contact Time [ms] (R)'] = df['Contact Time [ms] (R)'] * 1000

        metrics = ["Number of Hops/Reps", "Contact Time [ms]", "Impulse [N s]", "RSI (Flight/Contact Time)", 
                    "Mean Active Stiffness [N/m]", "Mean Average Power [W]", 
                    "Average Power Fatigue [W]", "Landing RFD Fatigue [%]", 
                    "RSI (Flight/Contact Time) Fatigue [%]", "Stiffness Fatigue [%]"
                ]
        for metric in metrics:
            left_col = f"{metric} (L)"
            right_col = f"{metric} (R)"
            asym_col = f"{metric} (Asym)(%)"
        
            # Calculate the combined result as the maximum of left and right
            df[metric] = np.maximum(df[left_col], df[right_col])
        
            # Calculate asymmetry using np.where for element-wise comparisons
            df[asym_col] = np.where(
                (df[left_col] < 0) & (df[right_col] > 0), 100.0,  # Left negative, right positive
                np.where(
                    (df[left_col] > 0) & (df[right_col] < 0), -100.0,  # Left positive, right negative
                    (np.minimum(df[left_col], df[right_col]) / df[metric] - 1) * 100  # Standard asymmetry calculation
                )
            )
        
            # Adjust for cases where right_col > left_col, only if asym_col is not -100.0
            df[asym_col] = np.where(
                (df[asym_col] != -100.0),  # Condition to avoid updating when asym_col is -100.0
                np.where(df[right_col] > df[left_col], abs(df[asym_col]), df[asym_col]),  # Adjust for right_col > left_col
                df[asym_col]  # If asym_col is -100.0, leave it as it is
)
        df = df.reindex(columns=columns_to_keep).round(2)
        return df

    def format_RSKIP(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {
                                    "weight": "BW [KG]",
                                    "testType": "Test Type",
                                    "Peak Vertical Force_Left": "Peak Vertical Force [N] (L)",
                                    "Peak Vertical Force_Right": "Peak Vertical Force [N] (R)",
                                    "Peak Vertical Force / BM_Left": "Peak Vertical Force / BM [N/kg] (L)",
                                    "Peak Vertical Force / BM_Right": "Peak Vertical Force / BM [N/kg] (R)",
                                    "Force at 100ms_Left": "Force at 100ms [N] (L)",
                                    "Force at 100ms_Right": "Force at 100ms [N] (R)",
                                    "Absolute Impulse - 100ms_Left": "Absolute Impulse - 100ms [N s] (L)",
                                    "Absolute Impulse - 100ms_Right": "Absolute Impulse - 100ms [N s] (R)"
                                })
        columns_to_keep = [
                            "Name",
                            "ExternalId",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps (L)",
                            "Reps (R)",
                            "Peak Vertical Force [N]",
                            "Peak Vertical Force [N] (L)",
                            "Peak Vertical Force [N] (R)",
                            "Peak Vertical Force [N] (Asym)(%)",
                            "Peak Vertical Force / BM [N/kg]",
                            "Peak Vertical Force / BM [N/kg] (L)",
                            "Peak Vertical Force / BM [N/kg] (R)",
                            "Peak Vertical Force / BM [N/kg] (Asym)(%)",
                            "Force at 100ms [N]",
                            "Force at 100ms [N] (L)",
                            "Force at 100ms [N] (R)",
                            "Force at 100ms [N] (Asym)(%)",
                            "Absolute Impulse - 100ms [N s]",
                            "Absolute Impulse - 100ms [N s] (L)",
                            "Absolute Impulse - 100ms [N s] (R)",
                            "Absolute Impulse - 100ms [N s] (Asym)(%)",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0
        metrics = [
                    "Peak Vertical Force [N]",
                    "Peak Vertical Force / BM [N/kg]",
                    "Force at 100ms [N]",
                    "Absolute Impulse - 100ms [N s]",
                ]

        for metric in metrics:
            left_col = f"{metric} (L)"
            right_col = f"{metric} (R)"
            asym_col = f"{metric} (Asym)(%)"
            df[metric] = np.where(
                df[left_col].isna(), 
                df[right_col], 
                np.where(
                    df[right_col].isna(), 
                    df[left_col], 
                    np.maximum(df[left_col], df[right_col])
                )
            )

        
            df[asym_col] = np.where(
                (df[left_col] < 0) & (df[right_col] > 0), 100.0,  # Left negative, right positive
                np.where(
                    (df[left_col] > 0) & (df[right_col] < 0), -100.0,  # Left positive, right negative
                    (np.minimum(df[left_col], df[right_col]) / df[metric] - 1) * 100  # Standard asymmetry calculation
                )
            )
        
            # Adjust for cases where right_col > left_col, only if asym_col is not -100.0
            df[asym_col] = np.where(
                (df[asym_col] != -100.0),  # Condition to avoid updating when asym_col is -100.0
                np.where(df[right_col] > df[left_col], abs(df[asym_col]), df[asym_col]),  # Adjust for right_col > left_col
                df[asym_col]  # If asym_col is -100.0, leave it as it is
        )
        df = df.reindex(columns=columns_to_keep).round(2)
        return df
        
    def format_SLSB(self, df):
        pst = pytz.timezone('America/Los_Angeles')
        df['adjusted_times'] = df['recordedDateUtc'].apply(lambda x: parser.parse(x).astimezone(pst))
        df['Date'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        df = df.rename(columns = {
                                "testType": "Test Type",
                                "weight": "BW [KG]",
                                "CoP Range - Medial-Lateral_Left": "CoP Range - Medial-Lateral [mm] (L)",
                                "CoP Range - Medial-Lateral_Right": "CoP Range - Medial-Lateral [mm] (R)",
                                "CoP Range - Anterior-Posterior_Left": "CoP Range - Anterior-Posterior [mm] (L)",
                                "CoP Range - Anterior-Posterior_Right": "CoP Range - Anterior-Posterior [mm] (R)",
                                "Mean Velocity_Left": "Mean Velocity [mm/s] (L)",
                                "Mean Velocity_Right": "Mean Velocity [mm/s] (R)",
                            })
        columns_to_keep = [
                            "Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps (L)",
                            "Reps (R)",
                            "Exercise Length [S]",
                            "Eyes Closed",
                            "Unstable Surface",
                            "Secondary Task",
                            "CoP Range - Medial-Lateral [mm]",
                            "CoP Range - Medial-Lateral [mm] (L)",
                            "CoP Range - Medial-Lateral [mm] (R)",
                            "CoP Range - Medial-Lateral [mm] (Asym)(%)",
                            "CoP Range - Anterior-Posterior [mm]",
                            "CoP Range - Anterior-Posterior [mm] (L)",
                            "CoP Range - Anterior-Posterior [mm] (R)",
                            "CoP Range - Anterior-Posterior [mm] (Asym)(%)",
                            "Mean Velocity [mm/s]",
                            "Mean Velocity [mm/s] (L)",
                            "Mean Velocity [mm/s] (R)",
                            "Mean Velocity [mm/s] (Asym)(%)",
                            "Tag",
                            'Tag Type'
                        ]
        for column in columns_to_keep:
            if column not in df.columns:
                df[column] = 0

        metrics = [
                    "CoP Range - Medial-Lateral [mm]",
                    "CoP Range - Anterior-Posterior [mm]",
                    "Mean Velocity [mm/s]"
                ]
        df['CoP Range - Medial-Lateral [mm] (L)'] = df['CoP Range - Medial-Lateral [mm] (L)'] * 1000
        df['CoP Range - Medial-Lateral [mm] (R)'] = df['CoP Range - Medial-Lateral [mm] (R)'] * 1000
        df['CoP Range - Anterior-Posterior [mm] (L)'] = df['CoP Range - Anterior-Posterior [mm] (L)'] * 1000
        df['CoP Range - Anterior-Posterior [mm] (R)'] = df['CoP Range - Anterior-Posterior [mm] (R)'] * 1000
        df['Mean Velocity [mm/s] (L)'] = df['Mean Velocity [mm/s] (L)'] * 1000
        df['Mean Velocity [mm/s] (R)'] = df['Mean Velocity [mm/s] (R)'] * 1000
        for metric in metrics:
            left_col = f"{metric} (L)"
            right_col = f"{metric} (R)"
            asym_col = f"{metric} (Asym)(%)"
            df[metric] = np.where(
                df[left_col].isna(), 
                df[right_col], 
                np.where(
                    df[right_col].isna(), 
                    df[left_col], 
                    np.maximum(df[left_col], df[right_col])
                )
            )

        
            # Calculate asymmetry using np.where for element-wise comparisons
            df[asym_col] = np.where(
                (df[left_col] < 0) & (df[right_col] > 0), 100.0,  # Left negative, right positive
                np.where(
                    (df[left_col] > 0) & (df[right_col] < 0), -100.0,  # Left positive, right negative
                    (np.minimum(df[left_col], df[right_col]) / df[metric] - 1) * 100  # Standard asymmetry calculation
                )
            )
        
            # Adjust for cases where right_col > left_col, only if asym_col is not -100.0
            df[asym_col] = np.where(
                (df[asym_col] != -100.0),  # Condition to avoid updating when asym_col is -100.0
                np.where(df[right_col] > df[left_col], abs(df[asym_col]), df[asym_col]),  # Adjust for right_col > left_col
                df[asym_col]  # If asym_col is -100.0, leave it as it is
        )
        columns_to_keep = [
                            "Name",
                            "Test Type",
                            "Date",
                            "Time",
                            "BW [KG]",
                            "Reps (L)",
                            "Reps (R)",
                            "Exercise Length [S]",
                            "Eyes Closed",
                            "Unstable Surface",
                            "Secondary Task",
                            "CoP Range - Medial-Lateral [mm]",
                            "CoP Range - Medial-Lateral [mm] (L)",
                            "CoP Range - Medial-Lateral [mm] (R)",
                            "CoP Range - Medial-Lateral [mm] (Asym)(%)",
                            "CoP Range - Anterior-Posterior [mm]",
                            "CoP Range - Anterior-Posterior [mm] (L)",
                            "CoP Range - Anterior-Posterior [mm] (R)",
                            "CoP Range - Anterior-Posterior [mm] (Asym)(%)",
                            "Mean Velocity [mm/s]",
                            "Mean Velocity [mm/s] (L)",
                            "Mean Velocity [mm/s] (R)",
                            "Mean Velocity [mm/s] (Asym)(%)",
                            "Tag",
                            'Tag Type'
                        ]

        df = df.reindex(columns=columns_to_keep).round(2)
        return df
    
    def parse_date_range(self, date_range):
        start_str, end_str = date_range.split('-')
        start_date = datetime.strptime(start_str, "%m/%d/%Y")
        end_date = datetime.strptime(end_str, "%m/%d/%Y")
        return start_date, end_date
    
    def generate_intervals(self, start_date, end_date, interval_days):
        intervals = []
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=interval_days)
            if current_end > end_date:
                current_end = end_date
            intervals.append((current_start, current_end - timedelta(seconds=1)))
            current_start = current_end
        return intervals

    def format_date_utc(self, date, is_start):
        if is_start:
            formatted = date.strftime("%Y-%m-%dT%H%%3A%M%%3A%SZ")
        else:
            end_time = date - timedelta(seconds=1)
            formatted = end_time.strftime("%Y-%m-%dT%H%%3A%M%%3A%SZ")
        return formatted
    
    def date_range_to_utc_intervals(self, date_range, interval_days):
        start_date, end_date = self.parse_date_range(date_range)
        intervals = self.generate_intervals(start_date, end_date, interval_days)
        utc_intervals = [(self.format_date_utc(start, True), self.format_date_utc(end, False)) for start, end in intervals]
        return utc_intervals

    def split_date_range_utc(self, start_str, end_str, fraction):

        start = urllib.parse.unquote(start_str)
        start = datetime.fromisoformat(start.replace('Z', '+00:00'))

        end = urllib.parse.unquote(end_str)
        end = datetime.fromisoformat(end.replace('Z', '+00:00'))

        total_duration = (end - start).total_seconds()
        interval_duration = total_duration * fraction

        intervals = []
        current_start = start

        while current_start < end:
            current_end = current_start + timedelta(seconds=interval_duration)
            if current_end > end:
                current_end = end

            start_iso = current_start.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_iso = current_end.strftime('%Y-%m-%dT%H:%M:%SZ')

            intervals.append(
                (
                    urllib.parse.quote(start_iso),
                    urllib.parse.quote(end_iso)
                )
            )
            current_start = current_end

        return intervals
    
    
    def fetch_data_recursively(self, start_date, end_date, granularity=1):
        intervals = self.split_date_range_utc(start_date, end_date, granularity)
        all_data = pd.DataFrame()
        print(intervals, "intervals(FF)")
        
        for start, end in intervals:
            data = self.get_tests((start.replace("%3A", ":"), end.replace("%3A", ":")))
            print(data, len(data), "appapap")
            print(start.replace("%3A", ":"), end.replace("%3A", ":"), len(data))
            if len(data) >= 50:
                smaller_data = self.fetch_data_recursively(start, end, granularity / 2)
                all_data = pd.concat([all_data, smaller_data])
            else:
                all_data = pd.concat([all_data, data])
        
        return all_data

    def get_data(self, date_range):
        start_date, end_date = date_range[0], date_range[1]
        df = self.fetch_data_recursively(start_date, end_date)
        if len(df) == 0:
            return None
        df = self.modify_df(df)
        #df = self.change_format(df)
        return df
    
    def update_forcedecks(self):

        last_update, last_index = self.get_last_update(self.vald_master_file_path)

        print(last_update, f"Fetching data from {last_update}")
        self.retrieve_tests_until_today(last_update)

        return

    def update_master_file(self, new_data):
        print(self.vald_master_file_path)
        directory = os.path.dirname(self.vald_master_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(self.vald_master_file_path):
            with open(self.vald_master_file_path, 'a') as f:
                f.write('\n')
            new_data.to_csv(self.vald_master_file_path, mode='a', header=False, index=True)
        else:
            new_data.to_csv(self.vald_master_file_path, index=True)
        
        print(f"Updated {self.vald_master_file_path}(FF)")


    
    def save_dataframes(self, teams_data):
        today_date = datetime.today().strftime('%Y-%m-%d')
        
        for team_name, test_data in teams_data.items():
            if not isinstance(team_name, str):
                team_name = str(team_name)
            
            sanitized_team_name = self.sanitize_foldername(team_name.lower())
            team_folder = os.path.join(self.base_directory, sanitized_team_name)
            if not os.path.exists(team_folder):
                os.makedirs(team_folder)

            forcedecks_folder = os.path.join(team_folder, 'ForceDecks')
            if not os.path.exists(forcedecks_folder):
                os.makedirs(forcedecks_folder)
            
            for test_type, df in test_data.items():
                existing_file_path = None
                
                
                search_pattern = f"{self.sanitize_filename(team_name.lower()).replace('/', '-')}_{test_type.lower().replace(' ', '_').replace('/', '-')}_*.csv"
                
                for file in os.listdir(forcedecks_folder):
                    if file.endswith('.csv') and file.startswith(search_pattern[:-5]):  # Strip off `*.csv`
                        existing_file_path = os.path.join(forcedecks_folder, file)
                        print(f"Existing file found: {existing_file_path}")
                        break
                
                if not existing_file_path:
                    print(f"No existing file found for {team_name} - {test_type}. Creating a new one.")
                else:
                    print(f"Appending to existing file: {existing_file_path}")

                # Save the new (or updated) dataframe, appending if file exists
                raw_file_name = f"{team_name}_{test_type}".replace('/', '-').lower()
                sanitized_file_name = self.sanitize_filename(raw_file_name) + '.csv'
                new_file_path = os.path.join(forcedecks_folder, sanitized_file_name)

                formatting_functions = {
                    'cmj': self.format_CMJ,
                    'abcmj': self.format_CMJ,
                    'slj': self.format_SLJ,
                    'sj': self.format_SJ,
                    'dj': self.format_DJ,
                    'lah': self.format_LAH,
                    'sllah': self.format_SLLAH,
                    'slsb': self.format_SLSB,
                    'hj': self.format_HJ,
                    'slhj': self.format_SLHJ,
                    'imtp': self.format_IMTP,
                    'isosqt': self.format_ISOSQT,                    
                    'rskip': self.format_RSKIP                    
                }
                
                # Convert test_type to lowercase
                test_type_lower = test_type.lower()
                
                # Check if the test_type exists in the mapping and call the corresponding function
                if test_type_lower in formatting_functions:
                    df = formatting_functions[test_type_lower](df)
                    print(f'Formatting {test_type} df')

                if os.path.exists(new_file_path):
                    df.to_csv(new_file_path, mode='a', header=False, index=False)
                    print(f"Appended data to {new_file_path}")
                else:
                    df.to_csv(new_file_path, index=False)
                    print(f"Created and saved new file {new_file_path}")



    def save_master_file(self, data):
        directory = os.path.dirname(self.vald_master_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data.to_csv(self.vald_master_file_path, index=True)

    def initial_setup(self):
        current_datetime = datetime.utcnow()

        if current_datetime < datetime(current_datetime.year, 8, 1): # start of athletic year - August 1
            year_for_august = current_datetime.year - 1
        else:
            year_for_august = current_datetime.year

        august_first_datetime = datetime(year_for_august, 8, 1)

        diff_in_months = (current_datetime.year - august_first_datetime.year) * 12 + current_datetime.month - august_first_datetime.month

        if abs(diff_in_months) > 6: # vald api cannot accept requests more than 6 months apart

            split_date = august_first_datetime + timedelta(days=180)
            split_date_formatted = split_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            august_first_formatted = august_first_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            one_hour_later_formatted = (current_datetime + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            formatted_dates_first = [august_first_formatted, split_date_formatted]
            formatted_dates_second = [split_date + timedelta(seconds=1), current_datetime]

            formatted_dates_second = [date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' if isinstance(date, datetime) else date for date in formatted_dates_second]

            print(formatted_dates_first, formatted_dates_second, "formatted_dates(FF)")

            data_first = self.get_data(formatted_dates_first)
            data_second = self.get_data(formatted_dates_second)
            
            data = pd.concat([data_first, data_second], ignore_index=True)
        else:
            august_first_formatted = august_first_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            one_hour_later_formatted = (current_datetime + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

            formatted_dates = [august_first_formatted, one_hour_later_formatted]

            data = self.get_data(formatted_dates)

        self.save_master_file(data)

        teams_data = self.data_to_groups(data)

        self.save_dataframes(teams_data)

    def retrieve_tests_until_today(self, start_date):    
        # List to store all tests data
        all_tests_data = []
    
        while True:  # Continue fetching until no more data is returned
            print(f"Fetching tests starting from: {start_date}")
    
            # Get the tests starting from the adjusted start_date
            tests_df, start_date = self.get_tests(start_date)
    
            if tests_df.empty:  # Exit if no data is returned
                print("No more tests available.")
                break  
    
            all_tests_data.append(tests_df)  # Append the fetched DataFrame
            print(f"Latest date updated to: {start_date}")
    
        if all_tests_data:
            combined_tests_df = pd.concat(all_tests_data, ignore_index=True)
            if os.path.exists(self.vald_master_file_path):
                old_data = pd.read_csv(self.vald_master_file_path)
                if 'testId' in combined_tests_df.columns and 'testId' in old_data.columns:
                    combined_tests_df = combined_tests_df[~combined_tests_df['testId'].isin(old_data['testId'])]
            self.update_master_file(combined_tests_df)
    
            teams_data = self.data_to_groups(combined_tests_df)
            self.save_dataframes(teams_data)

    def data_to_groups(self, data):
        teams_data = {}

        for group in data['Groups'].unique():
            teams_data[group] = {}
            group_data = data[data['Groups'] == group]
            
            for test in group_data['testType'].unique():
                test_data = group_data[group_data['testType'] == test].reset_index(drop=True)
                teams_data[group][test] = test_data

        return teams_data

    def populate_folders(self):
        if os.path.exists(self.vald_master_file_path) == False:
            print("Setting up intial(FD)")
            self.initial_setup()
        new_data = self.update_forcedecks()
        if new_data is None:
            return None
        self.update_master_file(new_data)
        teams_data = self.data_to_groups(new_data)

        self.save_dataframes(teams_data)


    def get_weight_table(self):
        if os.path.exists(self.vald_master_file_path) == False:
            print("No master file found")
            return
        master_df = pd.read_csv(self.vald_master_file_path)
        sports = master_df['Groups'].unique()
    
        # Define the date range
        date_range = pd.date_range(start='2023-08-01', end='2025-06-30')
    
        all_weight_tables = []
    
        for sport in sports:
            cmj_path = f'data/{sport}/ForceDecks/{sport}_cmj.csv'
            sj_path = f'data/{sport}/ForceDecks/{sport}_sj.csv'
    
            # Load weight data based on file availability
            if os.path.exists(cmj_path):
                print(f"Using CMJ weight for {sport}")
                weight_data = pd.read_csv(cmj_path)
            elif os.path.exists(sj_path):
                print(f"Using SJ weight for {sport}")
                weight_data = pd.read_csv(sj_path)
            else:
                print(f"No CMJ or SJ recorded for {sport}.")
                continue
    
            # Convert the 'Date' column to datetime
            weight_data['Date'] = pd.to_datetime(weight_data['Date'])
    
            athletes = weight_data['Name'].unique()
    
            # Create a complete date-Name grid
            weight_table = pd.DataFrame(
                [(athlete, date) for athlete in athletes for date in date_range],
                columns=['Name', 'Date']
            )
    
            # Map weights directly to the new weight table
            weight_table = weight_table.merge(
                weight_data[['Name', 'Date', 'BW [KG]']],
                on=['Name', 'Date'], how='left'
            )
    
            # Create a new column to store the filled weights
            weight_table['Weight'] = weight_table['BW [KG]']
    
            # Iterate over each athlete to fill weights
            for athlete in athletes:
                athlete_weights = weight_table[weight_table['Name'] == athlete]
    
                first_recorded_date = athlete_weights['Date'][athlete_weights['Weight'].notna()].min()
                
                weight_table['Weight'] = weight_table.groupby('Name')['BW [KG]'].ffill()
                weight_table['Weight'] = weight_table.groupby('Name')['Weight'].bfill()
    
            # Drop the original 'BW [KG]' column
            weight_table.drop(columns=['BW [KG]'], inplace=True)
    
            # Save the weight table for each sport
            output_path = f'data/{sport}/ForceDecks/{sport}_weight_table.csv'
            weight_table.to_csv(output_path, index=False)
            print(f"Saved weight table for {sport} at {output_path}.")
    
            # Store the weight table for optional further use
            all_weight_tables.append(weight_table)
    
        # Optional: Combine all weight tables if needed
        return pd.concat(all_weight_tables, ignore_index=True) if all_weight_tables else None

        