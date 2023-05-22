from  __future__ import  absolute_import
import time
from trainutils.averageMeter import AverageMeter
import torch
import torch.optim as optim
from pathlib import Path
from config.const import const_config


def get_optimizer(cfg,model):
    optimizer = None

    if cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=const_config.TRAIN.LR,
            momentum=const_config.TRAIN.MOMENTUM,
            weight_decay=const_config.TRAIN.WD,
            nesterov=const_config.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=const_config.TRAIN.LR,
        )
    elif cfg.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=const_config.TRAIN.LR,
            momentum=const_config.TRAIN.MOMENTUM,
            weight_decay=const_config.TRAIN.WD,
        )
    return optimizer

def create_log_folder(cfg):
    root_output_dir = Path(const_config.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    method = cfg.METHOD_DIR

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_output_dir / method / time_str / 'checkpoints'

    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / method / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

def train(train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = get_batch_label(dataset, idx)
        inp = inp.to(device)

        preds = model(inp).cpu()

        # 计算loss值
        batch_size = inp.size(0)
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % const_config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(val_loader, dataset, converter, model, criterion, device, epoch, writer_dict):
    losses = AverageMeter()
    model.eval()

    n_correct = 0
    sum = 0
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):

            labels = get_batch_label(dataset, idx)
            inp = inp.to(device)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                sum+=1
                if pred == target:
                    n_correct += 1

            if (i + 1) % const_config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == const_config.TEST.NUM_TEST:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:const_config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(const_config.TEST.NUM_TEST* const_config.TEST.BATCH_SIZE_PER_GPU)
    accuracy = n_correct / sum
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy