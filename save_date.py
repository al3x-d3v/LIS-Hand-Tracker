import csv

def save_on_csv(command, list):
    with open('command_data.csv', mode='a', newline="") as file :
        writer = csv.writer(file)
        writer.writerow([command] + list)