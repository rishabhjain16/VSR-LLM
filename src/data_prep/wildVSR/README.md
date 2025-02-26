Cleaning and processing wildVSR dataset for inference:

1. Download the dataset from [here](https://github.com/wildsr/wildsr-wildvrs-dataset)
2. Extract the dataset
3. Run the script to clean and process the dataset

```bash
python test_prep.py /path/to/wildVSR/dataset
```
This will create a test.tsv and test.wrd file in the output directory.

4. After running the script, you will have to run Clustering script to get the labels for the dataset. Follow the procedure method in [Clustering](../src/clustering/README.md)

4.1 Extract the features for the wildVSR dataset. Use the following script in clustering folder.

```bash
python dump_hubert_feature_video_only.py /home/rishabh/Desktop/Datasets/WildVSR/test_data/ test /home/rishabh/Desktop/Experiments/VSR-LLM/checkpoints/large_vox_iter5.pt 24 1 0 /home/rishabh/Desktop/Datasets/WildVSR/test_data/ --user_dir `pwd`/../
```

This will create test_0_1.len and test_0_1.npy files in the output directory.

4.2 We use the .km cluster already trained for the LRS3 dataset. Since wildVSR is only used for inference and doesn't contain enough data to train a new cluster, we use the dump_km_label.py script to dump the features for the wildVSR dataset using a clustering model trained on wither lrs2 or lrs3 dataset. This will create .len and .npy files for the wildVSR dataset in the output directory. 

```bash
python dump_km_label.py /home/rishabh/Desktop/Datasets/WildVSR/test_data/ test /home/rishabh/Desktop/Datasets/WildVSR/KM_clus/kmean_500_plus_n_20.km 1 0 /home/rishabh/Desktop/Datasets/WildVSR/test_data/
```
This creates test_0_1.km file in the output directory.

4.3 Now you can use the test_0_1.km file to get cluster labels for the wildVSR dataset. Open cluster_counts.py and replace unit_pth to path in `path_to_km_file.km`.\
Then you can get `.cluster_counts`.

4.4 Finally rename the .cluster_counts file to test.cluster_counts

5. Finally use can use test.cluster_counts, test.tsv and test.wrd file to run the inference.

