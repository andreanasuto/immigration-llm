import os
import pandas as pd

def aggregate_monthly_data(base_path):
    year_folders = [os.path.join(base_path, year) for year in os.listdir(base_path) if year.isdigit()]
    
    for year_folder in year_folders:
        if not os.path.isdir(year_folder):
            continue
        
        files_not_aggregated = []
        output_dir = os.path.join(year_folder, "aggregated")
        os.makedirs(output_dir, exist_ok=True)
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            output_file = os.path.join(output_dir, f"{month_str}_aggregated.csv")
            
            month_folder = os.path.join(year_folder, month_str)
            if not os.path.isdir(month_folder):
                continue
            
            file_batch = []
            
            for day in os.listdir(month_folder):
                day_folder = os.path.join(month_folder, day)
                if not os.path.isdir(day_folder):
                    continue
                
                for file in os.listdir(day_folder):
                    file_path = os.path.join(day_folder, file)
                    file_batch.append(file_path)
                    
                    if len(file_batch) == 10:
                        process_and_append_files(file_batch, output_file, files_not_aggregated)
                        file_batch = []
            
            if file_batch:
                process_and_append_files(file_batch, output_file, files_not_aggregated)
        
        log_file = os.path.join(year_folder, "files_not_aggregated.txt")
        with open(log_file, "w") as f:
            for item in files_not_aggregated:
                f.write(f"{item}\n")

def process_and_append_files(file_batch, output_file, files_not_aggregated):
    first_file = not os.path.exists(output_file)
    
    with open(output_file, 'a') as f_out:
        for file_path in file_batch:
            try:
                df = pd.read_csv(file_path, iterator=True, chunksize=10000)  # Process in chunks
                for chunk in df:
                    if 'date' in chunk.columns:
                        chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d %H:%M:%S')
                    if 'text' in chunk.columns:
                        chunk = chunk.drop(columns=['text'])
                    chunk.to_csv(f_out, index=False, header=first_file, mode='a')
                    first_file = False
            except Exception as e:
                files_not_aggregated.append(file_path)

base_path = "/n/netscratch/cga/Lab/anasuto/immigration/geotweets_usa"
aggregate_monthly_data(base_path)