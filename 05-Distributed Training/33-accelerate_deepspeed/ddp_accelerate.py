import time
import math
import torch
import pandas as pd

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from peft import LoraConfig, get_peft_model
from transformers import BertTokenizer, BertForSequenceClassification


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

    tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = BertForSequenceClassification.from_pretrained("/gemini/code/model")

    lora_config = LoraConfig(target_modules=["query", "key", "value"])

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=0.001)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, resume, epoch=3, log_step=10):
    global_step = 0
    start_time = time.time()

    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        resume_step = global_step = int(resume.split("step_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from checkpoint -> {resume}")

    for ep in range(resume_epoch, epoch):
        model.train()
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(trainloader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            active_dataloader = trainloader
        for batch in active_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.sync_gradients:
                    global_step += 1

                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, global_step)

                    if global_step % 50 == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save
                        )
        acc = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}, time: {time.time() - start_time}")
        accelerator.log({"acc": acc}, global_step)

    accelerator.end_training()


def main():

    accelerator = Accelerator(log_with="tensorboard", project_dir="ckpts")

    accelerator.init_trackers("runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)

    train(model, optimizer, trainloader, validloader, accelerator, resume=None)


if __name__ == "__main__":
    main()