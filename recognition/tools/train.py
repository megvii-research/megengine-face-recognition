import argparse
import bisect
import multiprocessing as mp
import os
import time

import megengine as mge
import megengine.autodiff as ad
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.optimizer as optim

from recognition.datasets import get_train_dataset
from recognition.models import FaceRecognitionModel
from recognition.tools.utils import AverageMeter, try_load_latest_checkpoint, load_config_from_path

logger = mge.get_logger(__name__)


def adjust_learning_rate(opt, epoch, configs):
    """adjust learning rate according to epoch. step learning rate scheduler is used according to configs

    Args:
        opt (optim.Optimizer): optimizer
        epoch (int): epoch
        configs (dict): configuration, required fields include:
                learning_rate: start learning rate
                learning_rate_gamma: by what factor learning rate decay each time
                learning_rate_milestons: epochs when learning rate decay
    """
    base_lr = configs["learning_rate"] * (
        configs["learning_rate_gamma"] ** bisect.bisect_right(configs["learning_rate_milestons"], epoch)
    )
    if dist.get_rank() == 0:
        logger.info(f"epoch {epoch}, using learning rate {base_lr:.3g}")
    for param_group in opt.param_groups:
        param_group["lr"] = base_lr


def main(args):
    configs = load_config_from_path(args.config_file)

    num_devices = dist.helper.get_device_count_by_fork("gpu")
    if num_devices > 1:
        # distributed training
        master_ip = "localhost"
        port = dist.get_free_ports(1)[0]
        dist.Server(port)
        processes = []
        for rank in range(num_devices):
            process = mp.Process(target=worker, args=(master_ip, port, num_devices, rank, configs))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
    else:
        # non-distributed training
        worker(None, None, 1, 0, configs)


def worker(master_ip, port, world_size, rank, configs):
    if world_size > 1:
        dist.init_process_group(
            master_ip=master_ip, port=port, world_size=world_size, rank=rank, device=rank,
        )
        logger.info("init process group for gpu{} done".format(rank))

    # set up logger
    os.makedirs(configs["base_dir"], exist_ok=True)
    worklog_path = os.path.join(configs["base_dir"], "worklog.txt")
    mge.set_log_file(worklog_path)

    # prepare model-related components
    model = FaceRecognitionModel(configs)

    # prepare data-related components
    preprocess = T.Compose([T.Normalize(mean=127.5, std=128), T.ToMode("CHW")])
    augment = T.Compose([T.RandomHorizontalFlip()])

    train_dataset = get_train_dataset(configs["dataset"], dataset_dir=configs["dataset_dir"])
    train_sampler = data.RandomSampler(train_dataset, batch_size=configs["batch_size"], drop_last=True)
    train_queue = data.DataLoader(train_dataset, sampler=train_sampler, transform=T.Compose([augment, preprocess]))

    # prepare optimize-related components
    configs["learning_rate"] = configs["learning_rate"] * dist.get_world_size()
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters())
        gm = ad.GradManager().attach(model.parameters(), callbacks=[dist.make_allreduce_cb("mean")])
    else:
        gm = ad.GradManager().attach(model.parameters())
    opt = optim.SGD(
        model.parameters(),
        lr=configs["learning_rate"],
        momentum=configs["momentum"],
        weight_decay=configs["weight_decay"],
    )

    # try to load checkpoint
    model, start_epoch = try_load_latest_checkpoint(model, configs["base_dir"])

    # do training
    def train_one_epoch():
        def train_func(images, labels):
            opt.clear_grad()
            with gm:
                loss, accuracy, _ = model(images, labels)
                gm.backward(loss)
                if dist.is_distributed():
                    # all_reduce_mean
                    loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
                    accuracy = dist.functional.all_reduce_sum(accuracy) / dist.get_world_size()
            opt.step()
            return loss, accuracy

        model.train()

        average_loss = AverageMeter("loss")
        average_accuracy = AverageMeter("accuracy")
        data_time = AverageMeter("data_time")
        train_time = AverageMeter("train_time")

        total_step = len(train_queue)
        data_iter = iter(train_queue)
        for step in range(total_step):
            # get next batch of data
            data_tic = time.time()
            images, labels = next(data_iter)
            data_toc = time.time()

            # forward pass & backward pass
            train_tic = time.time()
            images = mge.tensor(images, dtype="float32")
            labels = mge.tensor(labels, dtype="int32")
            loss, accuracy = train_func(images, labels)
            train_toc = time.time()

            # do the statistics and logging
            n = images.shape[0]
            average_loss.update(loss.item(), n)
            average_accuracy.update(accuracy.item() * 100, n)
            data_time.update(data_toc - data_tic)
            train_time.update(train_toc - train_tic)
            if step % configs["log_interval"] == 0 and dist.get_rank() == 0:
                logger.info(
                    "epoch: %d, step: %d, %s, %s, %s, %s",
                    epoch,
                    step,
                    average_loss,
                    average_accuracy,
                    data_time,
                    train_time,
                )

    for epoch in range(start_epoch, configs["num_epoch"]):
        adjust_learning_rate(opt, epoch, configs)
        train_one_epoch()

        if dist.get_rank() == 0:
            checkpoint_path = os.path.join(configs["base_dir"], f"epoch-{epoch+1}-checkpoint.pkl")
            mge.save(
                {"epoch": epoch + 1, "state_dict": model.state_dict()}, checkpoint_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", help="path to experiment configuration", required=True)
    args = parser.parse_args()

    main(args)
