0. Follow the steps in auto_avsr(https://github.com/mpc001/auto_avsr/tree/main/preparation) or in preparation folder (same as auto_avsr prep) until you get the folder with lrs2_video_seg24s.


1. Run: python lrs2_file_word_list_prep_step1.py 
Make sure to change the path in the python file to your newly created folder, It will create file.list and label.list in the destination folder. 

1.1. mv file.list, label.list to ...../path/lrs2/lrs2_video_seg24s/ folder
'''
mv file.list ./lrs2/lrs2_video_seg24s/
mv label.list ./lrs2/lrs2_video_seg24s/
'''
2.  python lrs2_cf_step2.py
- It will create count frames (.nframes similar to lrs3)

2.1. Also copy the files {test,val,train,pretrain}.txt to lrs2_video_seg24s folder

6. python lrs2_manifest_step3.py --lrs2 /data/ssd2/data_rishabh/lrs2/ --vocab-size 1000


