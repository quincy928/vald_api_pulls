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
        
        
        

        self.smartspeed_api_url = 'https://prd-use-api-extsmartspeed.valdperformance.com/v1/team/72f89fa3-989d-40c8-8769-37155a01259d/tests'
        self.groupnames_api_url = 'https://prd-use-api-externaltenants.valdperformance.com/groups'
        self.profiles_api_url = 'https://prd-use-api-externalprofile.valdperformance.com/profiles/'

        self.vald_master_file_path = os.path.join("data", "master_files", "smartspeed_allsports.csv")
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
        access_token = self.get_access_token()
        if not access_token:
            print("Failed to retrieve access token")
            return

        headers = {'Authorization': f'Bearer {access_token}', 'Accept': '*/*'}
        api_url = f"{self.smartspeed_api_url}?TestFromUtc={start_date}&Page={pageno}"
        print(f'The API URL is: {api_url}')
        tests_data = self.fetch_data(api_url, headers)
        if tests_data is None:
            return pd.DataFrame()
        api_url_groupnames = f"{self.groupnames_api_url}?TenantId={self.tenant_id}"
        group_data = self.fetch_data(api_url_groupnames, headers)
        id_to_name = {group['id']: group['name'] for group in group_data['groups']}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.fetch_data, f"{self.profiles_api_url}{test['profileId']}?TenantId={self.tenant_id}", headers) for test in tests_data]
            for test, future in zip(tests_data, futures):
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
                    print(f"An error occurred while processing test with profileId {test['profileId']}: {e}")
                    return pd.DataFrame()
        
        def flatten_json(nested_json, sep='.'):
            out = {}
            
            def flatten(x, name=''):
                if isinstance(x, dict):
                    for key, value in x.items():
                        flatten(value, name + key + sep)
                elif isinstance(x, list):
                    for i, value in enumerate(x):
                        flatten(value, name + str(i) + sep)
                else:
                    out[name[:-1]] = x
        
            flatten(nested_json)
            return out
        flattened_data = [flatten_json(record) for record in tests_data]
        

        df = pd.DataFrame(flattened_data)

        print("Data retrieval complete.")
        return df
    
    def modify_df(self, df):
        df['ExternalId'] = ""
        df['adjusted_times'] = df['testDateUtc'].apply(parser.parse)

        df['Date UTC'] = df['adjusted_times'].dt.strftime('%m/%d/%Y')
        df['Time UTC'] = df['adjusted_times'].dt.strftime('%I:%M %p')
        df.drop(columns=['adjusted_times'], inplace=True)
        
        df.rename(columns={'repCount': 'Rep Count',
                           'deviceCount': 'Device Count',
                           'runningSummaryFields.velocityFields.distance': 'Distance',
                           'runningSummaryFields.gateSummaryFields.splitOne': 'Split 1',
                           'runningSummaryFields.gateSummaryFields.splitTwo': 'Split 2',
                           'runningSummaryFields.gateSummaryFields.splitThree': 'Split 3',
                           'runningSummaryFields.gateSummaryFields.splitFour': 'Split 4',
                           'runningSummaryFields.gateSummaryFields.cumulativeOne': 'Cumulative 1',
                           'runningSummaryFields.gateSummaryFields.cumulativeTwo': 'Cumulative 2',
                           'runningSummaryFields.gateSummaryFields.cumulativeThree': 'Cumulative 3',
                           'runningSummaryFields.gateSummaryFields.cumulativeFour': 'Cumulative 4',
                           'runningSummaryFields.velocityFields.peakVelocityMetersPerSecond': 'Peak Velocity',
                           'runningSummaryFields.velocityFields.meanVelocityMetersPerSecond': 'Mean Velocity',
                           'runningSummaryFields.bestSplitSeconds': 'Best Split',
                           'runningSummaryFields.totalTimeSeconds': 'Total Time',
                           'runningSummaryFields.splitAverageSeconds': 'Average Split'},inplace = True)

        columns_to_keep = [
            'ExternalId', 'Name', 'Groups', 'Date UTC', 'Time UTC', 'testName', 'testTypeName', 'Rep Count', 'Device Count', 
            'Distance', 'Split 1', 'Split 2', 'Split 3', 'Split 4', 'Cumulative 1', 'Cumulative 2', 
            'Cumulative 3', 'Cumulative 4', 'Peak Velocity', 'Mean Velocity', 'Best Split', 
            'Total Time', 'Average Split', 'testDateUtc'
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
        print(intervals, "intervals")
        
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
        return df
    
    def update_smartspeed(self):

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
            print(f"Updated {self.vald_master_file_path}")
        except Exception as e:
            print(f"Error updating master file: {e}")

    def save_dataframes(self, teams_data):
        today_date = datetime.today().strftime('%Y-%m-%d')
        
        for team_name, test_data in teams_data.items():
            if not isinstance(team_name, str):
                team_name = str(team_name)
            
            sanitized_team_name = self.sanitize_foldername(team_name.lower())
            
    
            team_folder = os.path.join(self.base_directory, sanitized_team_name)
            if not os.path.exists(team_folder):
                os.makedirs(team_folder)
            
    
            smartspeed_folder = os.path.join(team_folder, 'SmartSpeed')
            if not os.path.exists(smartspeed_folder):
                os.makedirs(smartspeed_folder)
            
            for test_type, df in test_data.items():
                existing_file_path = None
                
    
                for file in os.listdir(smartspeed_folder):
                    if file.startswith(f"{self.sanitize_filename(team_name.lower()).replace('/', '-')}_{test_type.lower().replace(' ', '_').replace('/', '-')}_"):
                        existing_file_path = os.path.join(smartspeed_folder, file)
                        break
                
    
                if existing_file_path:
                    existing_df = pd.read_csv(existing_file_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                    os.remove(existing_file_path)
                
    
                raw_file_name = f"{team_name}_{test_type}_{today_date}".replace('/', '-').lower()
                sanitized_file_name = self.sanitize_filename(raw_file_name) + '.csv'
                new_file_path = os.path.join(smartspeed_folder, sanitized_file_name)
                
                df.to_csv(new_file_path, index=False)
                print(f"Saved {new_file_path}")

    def save_master_file(self, data):
        directory = os.path.dirname(self.vald_master_file_path)
        if not os.path.exists(directory):
            try:
                print(f"Directory to be created: {directory}")
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return

        try:
            data.to_csv(self.vald_master_file_path, index=True)
            print(f"Saved master file {self.vald_master_file_path}")
        except Exception as e:
            print(f"Error saving master file {self.vald_master_file_path}: {e}")

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

            print(formatted_dates_first, formatted_dates_second, "formatted_dates")

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

    def data_to_groups(self, data):
        teams_data = {}
        for group in data['Groups'].unique():
            teams_data[group] = {}
            group_data = data[data['Groups'] == group]
            
            for test in group_data['testName'].unique():
                test_data = group_data[group_data['testName'] == test].reset_index(drop=True)
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
                break
            
            # Concatenate the new tests to the master DataFrame
            new_data = pd.concat([new_data, new_tests], ignore_index=True)
            
            page += 1
        if os.path.exists(self.vald_master_file_path):
            old_data = pd.read_csv(self.vald_master_file_path)
            if 'id' in new_data.columns and 'id' in old_data.columns:
                print('Found duplicates, deleting')
                new_data = new_data[~new_data['id'].isin(old_data['id'])]
        
        new_data_formatted = self.modify_df(new_data)
        self.save_master_file(new_data_formatted)
        
        # Process the data into teams/groups
        teams_data = self.data_to_groups(new_data_formatted)
        
        # Save the processed team data
        self.save_dataframes(teams_data)


    def populate_folders(self):
        if os.path.exists(self.vald_master_file_path) == False:
            print("Setting up intial")
            self.initial_setup()
        new_data = self.update_smartspeed()
        if new_data is None:
            return None
        self.update_master_file(new_data)
        teams_data = self.data_to_groups(new_data)

        self.save_dataframes(teams_data)