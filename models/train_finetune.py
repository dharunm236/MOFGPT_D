# train_finetune.py
import sys
sys.path.append("../")
import torch
from torch.cuda.amp import autocast
import os
import numpy as np
from tqdm import tqdm
import glob
import json
import argparse
from torch.utils.data import DataLoader
import yaml
import wandb
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from dataset.dataset_gpt import MOF_ID_Dataset
from modules.model_utils import get_model
from modules.models import LLMModel, LLMFineTuneModel
from transformers import get_cosine_schedule_with_warmup
import pandas as pd


def check_data_leakage(train_data_list,
                       test_data_list):
    train_data_set = set([i[0] for i in train_data_list])
    test_data_set = set([i[0] for i in test_data_list])
    intersection = train_data_set.intersection(test_data_set)
    print(f"Number of intersection: {len(intersection)}")
    return len(intersection)

def split_csv(csv_filenames,
              train_test_ratio = 0.8,
              random_seed = 42):
    all_data_list = []

    for csv_filename in csv_filenames:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_data_list.append(row) 
    
    print(f"shape of all_csv before duplicate removal: {len(all_data_list)}")
    all_data_list = remove_duplicates(all_data_list)
    print(f"shape of all_csv after duplicate removal: {len(all_data_list)}")
    random.seed(random_seed)
    random.shuffle(all_data_list)
    split_idx = int(len(all_data_list)*train_test_ratio)
    train_data_list = all_data_list[:split_idx]
    test_data_list = all_data_list[split_idx:]
    print(f"shape of train_data_list: {len(train_data_list)}")
    print(f"shape of test_data_list: {len(test_data_list)}")
    return train_data_list, test_data_list

def train_one_epoch(llm_fine_tune_model,
                    train_dataloader,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    is_fp16,
                    logging_steps,
                    scaler,
                    model_name,
                    save_dir,
                    save_steps,
                    training_config,
                    loss_fn):
    llm_fine_tune_model.train()
    train_loss = 0
    total_count = 0

    loop = tqdm(train_dataloader,
                desc=f"Training epoch {epoch}",
                total=len(train_dataloader))
    for batch_idx, batch in enumerate(loop):
        token_ids = batch["token_ids"].to(device)
        mask_ids = batch["mask_ids"].to(device)
        y_energy = batch["label"].to(device)
        outputs = llm_fine_tune_model(token_ids=token_ids,
                                      mask_ids=mask_ids,)
        y = y_energy.reshape(-1)
        outputs = outputs.reshape(-1)
        loss = loss_fn(outputs, y)
        train_loss += loss.item() * len(y)
        total_count += len(y)

        if is_fp16:
            scaler.scale(loss).backward()
            if training_config["grad_clip_value"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(llm_fine_tune_model.parameters(),
                                               max_norm=training_config["grad_clip_value"])

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()


        if batch_idx % logging_steps == 0 and batch_idx != 0:
            loop.set_description(f"loss {train_loss/total_count:.4f}")
        loop.set_postfix(lr=optimizer.param_groups[0]["lr"])

        if batch_idx % save_steps == 0:
            save_dict = {"model_state_dict": llm_fine_tune_model.llm_network.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                         "epoch": epoch,
                         "loss": train_loss / total_count}
            torch.save(save_dict,
                       os.path.join(save_dir, f"{model_name}_latest.pt"))

    return [train_loss / total_count]


def eval_one_epoch(llm_fine_tune_model,
                   val_dataloader,
                   device,
                   epoch,
                   is_fp16,
                   model_name,
                   fine_tune_config,
                   loss_fn):
    llm_fine_tune_model.eval()
    val_loss = 0
    total_correct_preds = 0
    total_count = 0

    loop = tqdm(val_dataloader,
                desc=f"Evaluating epoch {epoch}",
                total=len(val_dataloader),
                colour="green")
    for batch_idx, batch in enumerate(loop):
        token_ids = batch["token_ids"].to(device)
        mask_ids = batch["mask_ids"].to(device)
        y_energy = batch["label"].to(device)
        y_topo = batch["topology_label"].to(device)
        with torch.no_grad():
            outputs = llm_fine_tune_model(token_ids=token_ids,
                                          mask_ids=mask_ids,)
        y = y_energy.reshape(-1)
        outputs = outputs.reshape(-1)
        loss = loss_fn(outputs, y)
        val_loss += loss.item() * len(y)
        total_count += len(y)
        loop.set_description(f"loss {val_loss / total_count:.4f}")
    return [val_loss / total_count]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--config_filename",
                      type=str,
                      default="../config/config_finetune.yaml",
                      help="Path to config file.")

    args = args.parse_args()

    with open(args.config_filename, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(config["model"]["model_config_filename"], "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    # extract different configs
    data_config = config["data"]
    tokenizer_config = data_config["tokenizer"]
    base_model_config = model_config["base_model"]
    training_config = config["training"]
    dataset_name = config["data"]["train_csv_filename"].split(
        "/train.csv")[0].split("/")[-1]
    print(f"Dataset name: {dataset_name}")
    config["project_name"] = f"mofgpt_{dataset_name}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.init(project=config["project_name"],
               config=config,
               name="Finetune")

    tokenizer = MOFTokenizerGPT(vocab_file=config["data"]["vocab_path"],
                                add_special_tokens=tokenizer_config["add_special_tokens"],
                                truncation=tokenizer_config["truncation"],
                                pad_token=tokenizer_config["pad_token"],
                                mask_token=tokenizer_config["mask_token"],
                                bos_token=tokenizer_config["bos_token"],
                                eos_token=tokenizer_config["eos_token"],
                                unk_token=tokenizer_config["unk_token"],
                                max_len=tokenizer_config["max_seq_len"],
                                use_topology=tokenizer_config["use_topology"],)

    base_model_config["vocab_size"] = tokenizer.vocab_size

    # loading topology labels file
    with open(data_config["topology_labels_map_filename"], "r") as f:
        topology_labels_map = json.load(f)

    train_data_pd = pd.read_csv(config["data"]["train_csv_filename"], header=None)
    val_data_pd = pd.read_csv(config["data"]["val_csv_filename"], header=None)
    test_data_pd = pd.read_csv(config["data"]["test_csv_filename"], header=None)

    train_data_np = train_data_pd.to_numpy()
    val_data_np = val_data_pd.to_numpy()
    test_data_np = test_data_pd.to_numpy()
    
    check_data_leakage(train_data_np.tolist(),
                       val_data_np.tolist())

    print(f"For train dataset, there are {len(train_data_np)} data.")
    train_dataset = MOF_ID_Dataset(data=train_data_np,
                                   tokenizer=tokenizer,
                                   ignore_index=data_config["ignore_index"],
                                   use_multiprocessing=data_config["use_multiprocessing"],
                                   topology_labels_map=topology_labels_map,)

    print(f"For val dataset, there are {len(val_data_np)} data.")
    val_dataset = MOF_ID_Dataset(data=val_data_np,
                                 tokenizer=tokenizer,
                                 ignore_index=data_config["ignore_index"],
                                 use_multiprocessing=data_config["use_multiprocessing"],
                                 topology_labels_map=topology_labels_map,)

    print(f"For test dataset, there are {len(test_data_np)} data.")
    test_dataset = MOF_ID_Dataset(data=test_data_np,
                                  tokenizer=tokenizer,
                                  ignore_index=data_config["ignore_index"],
                                  use_multiprocessing=data_config["use_multiprocessing"],
                                  topology_labels_map=topology_labels_map,)

    print(
        f"Total number of data: {len(train_data_np) + len(val_data_np) + len(test_data_np)} data.")
    print(
        f"Ratio of train data: {len(train_data_np)/(len(train_data_np) + len(val_data_np) + len(test_data_np))}")
    print(
        f"Ratio of val data: {len(val_data_np)/(len(train_data_np) + len(val_data_np) + len(test_data_np))}")
    print(
        f"Ratio of test data: {len(test_data_np)/(len(train_data_np) + len(val_data_np) + len(test_data_np))}")

    # creating dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=data_config["batch_size"],
                                  shuffle=True,
                                  num_workers=data_config["num_workers"],
                                  collate_fn=train_dataset.collate_fn,)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=data_config["batch_size"],
                                shuffle=False,
                                num_workers=data_config["num_workers"],
                                collate_fn=val_dataset.collate_fn,)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=data_config["batch_size"],
                                 shuffle=False,
                                 num_workers=data_config["num_workers"],
                                 collate_fn=test_dataset.collate_fn,)

    # creating model
    # adding special tokens to tokenizer
    base_model_config["pad_token_id"] = tokenizer.pad_token_id
    base_model_config["bos_token_id"] = tokenizer.bos_token_id
    base_model_config["eos_token_id"] = tokenizer.eos_token_id
    base_model_config["vocab_size"] = tokenizer.vocab_size
    base_model_config["ignore_index"] = data_config["ignore_index"]
    base_model_config["max_position_embeddings"] = tokenizer_config["max_seq_len"]

    # creating model
    model = get_model(base_model_config,
                      device,)
    fine_tune_config = model_config["fine_tune"]
    llm_fine_tune_model = LLMFineTuneModel(llm_network=model,
                                           llm_config=base_model_config,
                                           fine_tune_config=fine_tune_config,
                                           is_fp16=config["training"]["fp16"],).to(device)
    # breakpoint()
    if training_config["loss_fn"] == "mse":
        loss_fn = torch.nn.MSELoss()
    elif training_config["loss_fn"] == "mae":
        loss_fn = torch.nn.L1Loss()
    elif training_config["loss_fn"] == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()

    if training_config["do_inference"]:
        print("Doing inference.")
        saved_dict_for_inference = torch.load(training_config["saved_dict"])
        if llm_fine_tune_model.load_state_dict(saved_dict_for_inference["model_state_dict"]):
            print("Successfully loaded model for inference.")
        else:
            print("Failed to load model for inference.")
        llm_fine_tune_model.eval()
        test_returns = eval_one_epoch(llm_fine_tune_model,
                                      test_dataloader,
                                      device,
                                      0,
                                      is_fp16=training_config["fp16"],
                                      model_name=config["project_name"],
                                      fine_tune_config=fine_tune_config,
                                      loss_fn=loss_fn,)
        val_returns = eval_one_epoch(llm_fine_tune_model,
                                     val_dataloader,
                                     device,
                                     0,
                                     is_fp16=training_config["fp16"],
                                     model_name=config["project_name"],
                                     fine_tune_config=fine_tune_config,
                                     loss_fn=loss_fn,)
        print(f"Test loss: {test_returns[0]}")
        print(f"Val loss: {val_returns[0]}")
        return

    if base_model_config["resume_from_checkpoint"]:
        trained_dict = torch.load(base_model_config["pretrained_model_path"])
        if llm_fine_tune_model.llm_network.load_state_dict(trained_dict["model_state_dict"]):
            print("Successfully loaded pretrained model.")
        else:
            print("Failed to load pretrained model.")
    else:
        print("Training from scratch.")

    num_trainable_params = sum(
        p.numel() for p in llm_fine_tune_model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(num_trainable_params/1e6))

    # creating optimizer
    optimizer = torch.optim.AdamW(llm_fine_tune_model.parameters(),
                                  lr=training_config["optimizer"]["lr"],
                                  weight_decay=training_config["optimizer"]["weight_decay"],)

    # creating scheduler
    if training_config["scheduler"]["type"] == "cosine":
        num_training_steps = len(train_dataloader) * training_config["epochs"]
        num_warmup_steps = int(num_training_steps *
                               training_config["scheduler"]["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,)
        print("Using cosine scheduler with num_warmup_steps = {} and num_training_steps = {}.".format(num_warmup_steps,
                                                                                                      num_training_steps))

    elif training_config["scheduler"]["type"] == "none":
        scheduler = None
    # else:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,

        #    )
    if not os.path.exists(training_config["save_dir"]):
        os.makedirs(training_config["save_dir"])

    min_val_loss = np.inf
    training_config["logging_steps"] = int(
        len(train_dataloader) * training_config["logging_ratio"])
    training_config["save_steps"] = int(
        len(train_dataloader) * training_config["save_ratio"])
    print("Logging every {} steps.".format(training_config["logging_steps"]))

    # scaler for fp16
    scaler = torch.cuda.amp.GradScaler() if training_config["fp16"] else None

    for epoch in range(training_config["epochs"]):
        # train
        llm_fine_tune_model.train()
        train_returns = train_one_epoch(llm_fine_tune_model,
                                        train_dataloader,
                                        optimizer,
                                        scheduler,
                                        device,
                                        epoch,
                                        is_fp16=training_config["fp16"],
                                        logging_steps=training_config["logging_steps"],
                                        scaler=scaler,
                                        model_name=config["project_name"],
                                        save_dir=training_config["save_dir"],
                                        save_steps=training_config["save_steps"],
                                        training_config=training_config,
                                        loss_fn=loss_fn,)
        # eval
        llm_fine_tune_model.eval()
        val_returns = eval_one_epoch(llm_fine_tune_model,
                                     val_dataloader,
                                     device,
                                     epoch,
                                     is_fp16=training_config["fp16"],
                                     model_name=config["project_name"],
                                     fine_tune_config=fine_tune_config,
                                     loss_fn=loss_fn,)
        wandb.log({"train_epoch_loss": train_returns[0],
                   "val_epoch_loss": val_returns[0],
                   "epoch": epoch, 
                   })
        print("Epoch {} train loss: {:.4f} val loss: {:.4f}".format(
            epoch, train_returns[0], val_returns[0]))

        if val_returns[0] < min_val_loss:
            min_val_loss = val_returns[0]
            save_dict = {"model_state_dict": llm_fine_tune_model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                         "epoch": epoch,
                         "loss": val_returns[0]}
            torch.save(save_dict,
                       os.path.join(training_config["save_dir"], f"{config['project_name']}_best.pt"))
            print("Saved best model at epoch {} with val loss {:.4f}.".format(
                epoch, val_returns[0]))


if __name__ == "__main__":
    main()
