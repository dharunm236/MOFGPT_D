import numpy as np
import pandas as pd
import os
import shutil


def check_duplicates(test, val):
  test_set = set(test.iloc[:,0])
  val_set = set(val.iloc[:,0])
  intersection = test_set.intersection(val_set)
  if len(intersection) > 0:
    print(f"Intersection of test and val: {intersection}")
    return False
  return True

def split():
    save_path = "../benchmark_datasets/finetune"
    val_test_ratio = 0.25 # so total is 0.80:0.05:0.15 for train:val:test
    filenames = ['../benchmark_datasets/hMOF_CH4_0.5_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CH4_0.05_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CH4_0.9_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CH4_2.5_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CH4_4.5_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CO2_0.1_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CO2_0.01_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CO2_0.5_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CO2_2.5_small_mofid_finetune',
                 '../benchmark_datasets/hMOF_CO2_0.05_small_mofid_finetune']
    for filename in filenames:
        train = pd.read_csv(filename + '/train.csv')
        test = pd.read_csv(filename + '/test.csv')
        val = test.sample(frac=val_test_ratio)
        test = test.drop(val.index)

        print(f"f{filename} train: {len(train)}, val: {len(val)}, test: {len(test)}")
        res = check_duplicates(test, val)
        print(f"No Duplicates found: {res}")
        # break
        if res:
            if os.path.exists(f"{save_path}/{filename.split('/')[-1]}"):
                shutil.rmtree(f"{save_path}/{filename.split('/')[-1]}")
            os.makedirs(f"{save_path}/{filename.split('/')[-1]}")
            train.to_csv(f"{save_path}/{filename.split('/')[-1]}/train.csv", index=False)
            val.to_csv(f"{save_path}/{filename.split('/')[-1]}/val.csv", index=False)
            test.to_csv(f"{save_path}/{filename.split('/')[-1]}/test.csv", index=False)
            print(f"Saved to {save_path}/{filename.split('/')[-1]}")
            train_saved = pd.read_csv(f"{save_path}/{filename.split('/')[-1]}/train.csv")
            val_saved = pd.read_csv(f"{save_path}/{filename.split('/')[-1]}/val.csv")
            test_saved = pd.read_csv(f"{save_path}/{filename.split('/')[-1]}/test.csv")
            print(f"Saved train: {len(train_saved)}, val: {len(val_saved)}, test: {len(test_saved)}")
      
if __name__ == "__main__":
    split()
