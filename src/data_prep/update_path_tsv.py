import csv
import sys

def update_file_paths(input_file, output_file, old_path, new_path):
    with open(input_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        rows = list(reader)

    updated_rows = []
    for row in rows:
        updated_row = [col.replace(old_path, new_path) for col in row]
        updated_rows.append(updated_row)

    with open(output_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerows(updated_rows)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python update_paths.py <input_file> <output_file> <old_path> <new_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    old_path = sys.argv[3]
    new_path = sys.argv[4]

    update_file_paths(input_file, output_file, old_path, new_path)


## Usage : python update_paths.py input.tsv output.tsv /home/rishabh/Desktop/Dataset/ /home/rjain/data/lrs3/433h_data