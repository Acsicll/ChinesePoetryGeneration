import time
import logging
import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from SongIambicsGeneration.utils import generate_model_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    scaler = torch.amp.GradScaler(enabled=True)
    for i, batch in enumerate(iterator):
        src, trg = batch
        src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=True):
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def to_train(model_type,epoch, model, train_loader,learning_rate, batch_size, optimizer, criterion, clip, device):
    cudnn.benchmark = True
    cudnn.enabled = True
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    if isinstance(epoch, list) and len(epoch) == 2:
        start_epoch, end_epoch = epoch
    elif isinstance(epoch, int):
        start_epoch, end_epoch = epoch
    else:
        raise ValueError("epoch must be int or list of int")

    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] start training...")
    for i in range(start_epoch, end_epoch):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        scheduler.step(train_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        logging.info(f'Epoch: {start_epoch + 1:02} | Train Loss: {train_loss:.3f} | Time: {epoch_mins}m {epoch_secs}s')

        model_name = generate_model_name("seq2seq","github",model_type,learning_rate,
                                         batch_size,i,train_loss)
        torch.save(model.state_dict(), f'./SavedModels/{model_name}')
        logging.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] Model saved as models./SavedModels/seq2seq_{i}.pt")

    print("Training complete!")
