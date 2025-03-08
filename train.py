import os.path
import time
import logging
import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint

from SongIambicsGeneration.utils import generate_model_name


def forward_with_checkpointing(model, src, trg, teacher_forcing_ratio):
    return checkpoint(model, src, trg, teacher_forcing_ratio, use_reentrant=False)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            src, trg = batch
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                output = model(src, trg, 0)  # turn off teacher forcing
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)

            val_loss += loss.item()
    return val_loss / len(val_loader)


def train(model, train_loader, optimizer, criterion, clip, scaler, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    # scaler = torch.amp.GradScaler(enabled=True)
    for i, batch in enumerate(train_loader):
        src, trg = batch
        src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=True):
            output = forward_with_checkpointing(model, src, trg, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def to_train(model_type, epoch, model, train_loader, val_loader, learning_rate, batch_size, optimizer, criterion, clip,
             device):
    if isinstance(epoch, list) and len(epoch) == 3:
        start_epoch, end_epoch, step = epoch
    elif isinstance(epoch, int):
        start_epoch, end_epoch, step = 0, epoch, 1
    else:
        raise ValueError("epoch must be int or list of int")

    cudnn.benchmark = True
    model.to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # 2
    scaler = torch.amp.GradScaler(enabled=True)

    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] start training...")
    for i in range(start_epoch, end_epoch):
        start_time = time.time()

        if True:
            #teacher_forcing_ratio = max(0.5 - (i * 0.05), 0.1)
            teacher_forcing_ratio = 0.5

        train_loss = train(model, train_loader, optimizer, criterion, clip, scaler, teacher_forcing_ratio, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if i % 2 == 0 and i != 0:
            torch.cuda.empty_cache()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        model_name = generate_model_name("seq2seq", "github", model_type, learning_rate,
                                         batch_size, i, train_loss)
        if not os.path.isdir(f'./SavedModels/{time.strftime("%Y%m%d", time.localtime(time.time()))}'):
            os.makedirs(f'./SavedModels/{time.strftime("%Y%m%d", time.localtime(time.time()))}')
        torch.save(model.state_dict(), f'./SavedModels/{time.strftime("%Y%m%d", time.localtime(time.time()))}/{model_name}')
        logging.info(
            f'[ Epoch: {i + 1:02} | Val Loss: {val_loss:.3f} |Train Loss: {train_loss:.3f} | Time: {epoch_mins}m {epoch_secs}s'
            f'| {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}  Model saved as ./SavedModels/{model_name} ]')

    print("Training complete!")
