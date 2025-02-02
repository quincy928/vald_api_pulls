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
from dotenv import load_dotenv

load_dotenv()


class Vald():
    def __init__(self):
        self.token_url = 'https://security.valdperformance.com/connect/token'
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.tenant_id = os.getenv("TENANT_ID")

        #self.modified_from_utc = os.getenv("MODIFIED_FROM_UTC")
        
        
        

        self.nordbord_api_url = 'https://prd-use-api-externalnordbord.valdperformance.com/tests'
        self.groupnames_api_url = 'https://prd-use-api-externaltenants.valdperformance.com/groups'
        self.profiles_api_url = 'https://prd-use-api-externalprofile.valdperformance.com/profiles/'

        self.vald_master_file_path = os.path.join("data", "master_files", "nordbord_allsports.csv")
        self.base_directory = 'data'
    
    def get_last_update(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_row = df.iloc[-1]
        test_date_utc = last_row['testDateUtc']
        last_index = last_row.name

        test_date_dt = datetime.strptime(test_date_utc, "%Y-%m-%dT%H:%M:%S.%fZ")

        updated_test_date_dt = test_date_dt + timedelta(milliseconds=1)
        updated_test_date_utc = updated_test_date_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        
        return updated_test_date_utc, last_index
    
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
    
    def get_tests(self, start_date, pageno):
        print(f'Getting tests starting from {start_date} on page number {pageno}')
        current_datetime = datetime.utcnow()
        current_formatted = current_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        access_token = self.get_access_token()
        if not access_token:
            print("Failed to retrieve access token(NB)")
            return

        headers = {'Authorization': f'Bearer {access_token}', 'Accept': '*/*'}
        api_url = f"{self.nordbord_api_url}?TenantId={self.tenant_id}&ModifiedFromUtc={start_date}&TestFromUtc={start_date}&TestToUtc={current_formatted}&Page={pageno}"

        tests_data = self.fetch_data(api_url, headers)
        if tests_data is None:
            return pd.DataFrame()
        api_url_groupnames = f"{self.groupnames_api_url}?TenantId={self.tenant_id}"
        group_data = self.fetch_data(api_url_groupnames, headers)
        id_to_name = {group['id']: group['name'] for group in group_data['groups']}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.fetch_data, f"{self.profiles_api_url}{test['athleteId']}?TenantId={self.tenant_id}", headers) for test in tests_data['tests']]
            for test, future in zip(tests_data['tests'], futures):
                try:
                    result = future.result()
                    if result is not None:
                        test['Name'] = result['givenName'].strip() + " " + result['familyName'].strip()
                        group_ids = result['groupIds']
                        group_names = [id_to_name.get(g_id, "ID not found") for g_id in group_ids]
                        test['Groups'] = '|'.join(group_names)
                    else:
                        continue
                        #return pd.DataFrame()
                except Exception as e:
                    print(f"An error occurred while processing test with athleteId {test['athleteId']}: {e}(NB)")
                    return pd.DataFrame()

        print("Data retrieval complete.(NB)")
        return pd.json_normalize(tests_data['tests'])
    
    def modify_df(self, df):
        df['ExternalId'] = ""
        df['adjusted_times'] = df['testDateUtc'].apply(parser.parse)

        df['Date UTC'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time UTC'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        
        df.rename(columns={'device': 'Device',
                           'Name' : 'Athlete',
                           'Date UTC' : 'Date',
                           'testTypeName': 'Test',
                           'notes': 'Notes',
                           'leftRepetitions': 'L Reps',
                           'rightRepetitions': 'R Reps',
                           'leftMaxForce': 'L Max Force (N)',
                           'rightMaxForce': 'R Max Force (N)',
                           'leftTorque': 'L Max Torque (Nm)',
                           'rightTorque': 'R Max Torque (Nm)',
                           'leftAvgForce': 'L Avg Force (N)',
                           'rightAvgForce': 'R Avg Force (N)',
                           'leftImpulse': 'L Max Impulse (Ns)',
                           'rightImpulse': 'R Max Impulse (Ns)'}, inplace=True)
        
        df['Max Imbalance (%)'] = (df['R Max Force (N)'] - df['L Max Force (N)']) / df[['R Max Force (N)', 'L Max Force (N)']].max(axis=1) * 100
        df['Avg Imbalance (%)'] = (df['R Avg Force (N)'] - df['L Avg Force (N)']) / df[['R Avg Force (N)', 'L Avg Force (N)']].max(axis=1) * 100
        df['Impulse Imbalance (%)'] = (df['R Max Impulse (Ns)'] - df['L Max Impulse (Ns)']) / df[['R Max Impulse (Ns)', 'L Max Impulse (Ns)']].max(axis=1) * 100
        
        reorder = ['athleteId', 'testId', 'modifiedUtc', 'testDateUtc', 'testTypeId', 'Athlete', 'Groups', 'ExternalId', 'Date', 'Time UTC', 'Device', 'Test',
       'L Reps', 'R Reps', 'L Max Force (N)', 'R Max Force (N)',
       'Max Imbalance (%)', 'L Max Torque (Nm)', 'R Max Torque (Nm)',
       'L Avg Force (N)', 'R Avg Force (N)', 'Avg Imbalance (%)',
       'L Max Impulse (Ns)', 'R Max Impulse (Ns)', 'Impulse Imbalance (%)', 'rightCalibration', 'leftCalibration',
       'Notes']
        df = df[reorder]
        return df

        
    def update_nordbord(self):

        last_update, last_index = self.get_last_update(self.vald_master_file_path)
        print(f'Last updated at {last_update}')
        self.get_data_until_today(last_update)

    def update_master_file(self, new_data):
        try:
            if os.path.exists(self.vald_master_file_path):
                with open(self.vald_master_file_path, 'a') as f:
                    f.write('\n')
                new_data.to_csv(self.vald_master_file_path, mode='a', header=False, index=True)
            else:
                new_data.to_csv(self.vald_master_file_path, index=True)
            print(f"Updated {self.vald_master_file_path}(NB)")
        except Exception as e:
            print(f"Error updating master file: {e}(NB)")

    def save_dataframes(self, teams_data):
        today_date = datetime.today().strftime('%Y-%m-%d')
        
        for team_name, test_data in teams_data.items():
            if not isinstance(team_name, str):
                team_name = str(team_name)
            
            sanitized_team_name = self.sanitize_foldername(team_name.lower())
            team_folder = os.path.join(self.base_directory, sanitized_team_name)
            if not os.path.exists(team_folder):
                os.makedirs(team_folder)

            nordbord_folder = os.path.join(team_folder, 'NordBord')
            if not os.path.exists(nordbord_folder):
                os.makedirs(nordbord_folder)
            
            for test_type, df in test_data.items():
                existing_file_path = None

                # Update: Include file extension in matching
                search_pattern = f"{self.sanitize_filename(team_name.lower()).replace('/', '-')}_{test_type.lower().replace(' ', '_').replace('/', '-')}_*.csv"
                
                for file in os.listdir(nordbord_folder):
                    if file.endswith('.csv') and file.startswith(search_pattern[:-5]):  # Strip off `*.csv`
                        existing_file_path = os.path.join(nordbord_folder, file)
                        print(f"Existing file found: {existing_file_path}")
                        break
                
                if not existing_file_path:
                    print(f"No existing file found for {team_name} - {test_type}. Creating a new one.")
                else:
                    print(f"Appending to existing file: {existing_file_path}")

                # Save the new (or updated) dataframe, appending if file exists
                raw_file_name = f"{team_name}_{test_type}".replace('/', '-').lower()
                sanitized_file_name = self.sanitize_filename(raw_file_name) + '.csv'
                new_file_path = os.path.join(nordbord_folder, sanitized_file_name)
                
                # Append to the CSV if it exists, otherwise create a new one
                if os.path.exists(new_file_path):
                    df.to_csv(new_file_path, mode='a', header=False, index=False)
                    print(f"Appended data to {new_file_path}")
                else:
                    df.to_csv(new_file_path, index=False)
                    print(f"Created and saved new file {new_file_path}")


    def save_master_file(self, data):
        directory = os.path.dirname(self.vald_master_file_path)
        if not os.path.exists(directory):
            try:
                print(f"Directory to be created: {directory}(NB)")
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}(NB)")
                return

        try:
            data.to_csv(self.vald_master_file_path, index=True)
            print(f"Saved master file {self.vald_master_file_path}(NB)")
        except Exception as e:
            print(f"Error saving master file {self.vald_master_file_path}: {e}(NB)")

    
    def data_to_groups(self, data):
        teams_data = {}
        for group in data['Groups'].unique():
            teams_data[group] = {}
            group_data = data[data['Groups'] == group]
            
            for test in group_data['Test'].unique():
                test_data = group_data[group_data['Test'] == test].reset_index(drop=True)
                teams_data[group][test] = test_data

        return teams_data

    def get_data_until_today(self, start_date):
        page = 1
        new_data = pd.DataFrame()  # Initialize an empty DataFrame
        
        while True:
            # Fetch the new batch of tests
            new_tests = self.get_tests(start_date, page)
            
            # If no new tests are found, break the loop
            if new_tests.empty:
                print('No new tests were found, the master file is up to date.')
                break
            
            # Concatenate the new tests to the master DataFrame
            new_data = pd.concat([new_data, new_tests], ignore_index=True)
            
            page += 1
        if os.path.exists(self.vald_master_file_path):
            old_data = pd.read_csv(self.vald_master_file_path)
            if 'id' in new_data.columns and 'id' in old_data.columns:
                duplicates = new_data[new_data['id'].isin(old_data['id'])]
                
                if not duplicates.empty:
                    print('Found duplicates, deleting:')
                    duplicates_info = duplicates[['id', 'Name', 'Groups', 'testDateUtc']]
                    print(duplicates_info.head(5))
                    if len(duplicates) > 5:
                        print(f"... and {len(duplicates) - 5} more duplicate records.")
                    new_data = new_data[~new_data['id'].isin(old_data['id'])]
        
        new_data_formatted = self.modify_df(new_data)
        self.save_master_file(new_data_formatted)
        
        # Process the data into teams/groups
        teams_data = self.data_to_groups(new_data_formatted)
        
        # Save the processed team data
        self.save_dataframes(teams_data)

