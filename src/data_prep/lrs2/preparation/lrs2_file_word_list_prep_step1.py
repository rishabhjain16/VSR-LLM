import os
import pandas as pd
from tqdm import tqdm

def get_file_label(lrs2_root):
    video_ids_total, labels_total = [], []
    csv_files = {
        'train': 'lrs2_train_transcript_lengths_seg24s.csv',
        'val': 'lrs2_val_transcript_lengths_seg24s.csv',
        'test': 'lrs2_test_transcript_lengths_seg24s.csv'
    }

    for split, csv_file in csv_files.items():
        # Read the CSV file
        csv_file_path = os.path.join(lrs2_root, 'labels', csv_file)
        df = pd.read_csv(csv_file_path, header=None)

        # Process each row in the CSV file
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            video_file = row[1]
            parts = video_file.split('/')
            video_id = parts[-2]
            file_name = parts[-1].replace('.mp4', '')

            # Determine if the path belongs to main or pretrain based on file structure
            if 'pretrain' in video_file:
                relative_path = f"pretrain/{video_id}/{file_name}"
                txt_path = os.path.join(lrs2_root, 'lrs2', 'lrs2_text_seg24s', 'pretrain', video_id, f'{file_name}.txt')
                txt_path_with_suffix = txt_path.replace('.txt', '_00.txt')
                if os.path.exists(txt_path_with_suffix):
                    txt_path = txt_path_with_suffix
            else:
                relative_path = f"main/{video_id}/{file_name}"
                txt_path = os.path.join(lrs2_root, 'lrs2', 'lrs2_text_seg24s', 'main', video_id, f'{file_name}.txt')

            # Check if the text file exists and read the label
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as txt_file:
                    label = txt_file.readlines()[0].strip()
                    labels_total.append(label)
                    video_ids_total.append(relative_path)
            else:
                print(f"Skipping {video_file}: {txt_path} does not exist.")

    video_id_fn = os.path.join(lrs2_root, 'file.list')
    label_fn = os.path.join(lrs2_root, 'label.list')
    print(video_id_fn, label_fn)
    with open(video_id_fn, 'w') as fo:
        fo.write('\n'.join(video_ids_total)+'\n')
    with open(label_fn, 'w') as fo:
        fo.write('\n'.join(labels_total)+'\n')
    return

# Main function
def main():
    # Set the root directory for LRS2 dataset
    lrs2_root = '/data/ssd2/data_rishabh/lrs2/segmented/'

    # Run the function to generate file.list and label.list
    get_file_label(lrs2_root)

if __name__ == "__main__":
    main()
