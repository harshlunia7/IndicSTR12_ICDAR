# Convolutional Recurrent Neural Network + CTCLoss | STAR-Net

## Dependence

- Python3.6.5
- torch==1.2.0
- torchvision==0.4.0
- tensorboard==2.3.0

## Train your data

### Prepare data

- Follow the instructions in [meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>) to create lmdb datasets
	Use the same step to create train and val data.

### Change parameters and alphabets

Parameters and alphabets can't always be the same in different situation. 

- Change parameters in the mytrain.py file
- Change alphabets

  Please put all the alphabets appeared in your labels in a file and input as charlist to the mytrain.py, or the program will throw error during training process.

### Train

Run `mytrain.py` by

```sh
python3 mytrain.py --trainRoot /ssd_scratch/cvit/sanjana/hindi-train-lmdb \
--valRoot /ssd_scratch/cvit/sanjana/hindi-test-lmdb \
--arch crnn --lan hindi --charlist /ssd_scratch/cvit/sanjana/crnn_new/lexicon.txt \
--batchSize 32 --nepoch 15 --cuda --expr_dir /ssd_scratch/cvit/sanjana \
--displayInterval 10 --valInterval 100 --adadelta \ 
--manualSeed 1234 --random_sample --deal_with_lossnan 
```

## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>)
[Sierkinhane/crnn_chinese_characters_rec](<https://github.com/Sierkinhane/crnn_chinese_characters_rec>)
