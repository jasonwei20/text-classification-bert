from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import BertForSequenceClassification
from transformers import BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import common, configuration, dataloader, utils_torch, visualization

def initialize_model(
    cfg,
    train_dataloader,
):
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = cfg.num_output_classes, 
        output_attentions = False,
        output_hidden_states = False,
    )
    
    model.cuda(); print('\n\n')

    optimizer = AdamW(model.parameters(), lr = 2e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps = len(train_dataloader) * cfg.num_epochs
    )

    device = utils_torch.get_device()

    return model, optimizer, scheduler, device

def evaluate_model(
    model,
    device,
    test_dataloader,
):
    model.eval()
    
    val_preds_list = []; val_gt_list = []

    for mb in test_dataloader:

        input_ids = mb[0].to(device)
        input_mask = mb[1].to(device)
        labels = mb[2].to(device)

        with torch.no_grad():

            val_loss, logits = model(input_ids, attention_mask=input_mask, labels=labels)
            val_confs, val_preds = torch.max(logits, dim=1)

            val_preds = val_preds.detach().cpu().numpy()
            val_gt = labels.to('cpu').numpy()
            val_preds_list.append(val_preds)
            val_gt_list.append(val_gt)

    val_preds_all = np.concatenate(val_preds_list, axis=None)
    val_gt_all = np.concatenate(val_gt_list, axis=None)
    val_acc = accuracy_score(val_gt_all, val_preds_all)

    return val_acc

def vanilla_finetune_bert(
    cfg,
    train_dataloader,
    test_dataloader,
):

    model, optimizer, scheduler, device = initialize_model(cfg, train_dataloader)
    train_loss_list = []; val_acc_list = []

    for epoch_num in range(1, cfg.num_epochs + 1):

        model.train()
        iter_bar = tqdm(train_dataloader)

        for mb_num, mb in enumerate(iter_bar):

            input_ids = mb[0].to(device)
            input_mask = mb[1].to(device)
            labels = mb[2].to(device)

            model.zero_grad()
            train_loss, logits = model(input_ids, attention_mask=input_mask, labels=labels)
            train_confs, train_preds = torch.max(logits, dim=1)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            mb_train_acc = accuracy_score(labels.cpu(), train_preds.cpu())
            val_acc = evaluate_model(model, device, test_dataloader)

            iter_bar_str =  (f"epoch {epoch_num}/{cfg.num_epochs}, " 
                            + f"mb {mb_num + 1}/{len(iter_bar)}: " 
                            + f"mb_train_loss={float(train_loss):.4f}, "  
                            + f"mb_train_acc={float(mb_train_acc):.3f}, "  
                            + f"val_acc={float(val_acc):.3f}")
            iter_bar.set_description(iter_bar_str)

            train_loss_list.append(train_loss); val_acc_list.append(val_acc)
    
    Path(f"plots/{cfg.exp_id}").mkdir(parents=True, exist_ok=True)
    visualization.plot_jasons_lineplot(None, train_loss_list, 'updates', 'training loss', f"{cfg.train_path.split('/')[-2]} n_train={cfg.train_subset}", f"plots/{cfg.exp_id}/train_loss.png")
    visualization.plot_jasons_lineplot(None, val_acc_list, 'updates', 'validation accuracy', f"{cfg.train_path.split('/')[-2]} n_train={cfg.train_subset} max_val_acc={max(val_acc_list):.3f}", f"plots/{cfg.exp_id}/val_acc.png")
