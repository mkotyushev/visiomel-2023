import collections
import logging
import multiprocessing
import os
import sys
import wandb

from src.utils.utils import MyLightningCLI, TrainerWandb


# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format = '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s'
)


Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("fold_index", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("score"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.fold_index)
    config = worker_data.config
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )

    cv_args = ['--data.init_args.fold_index', f'{worker_data.fold_index}']
    score = float('-inf')
    try:
        cli = MyLightningCLI(
            trainer_class=TrainerWandb, 
            save_config_kwargs={
                'config_filename': 'config_pl.yaml',
                'overwrite': True,
            },
            args=sys.argv[1:] + cv_args,
            run=True,
        )
        score = cli.trainer.checkpoint_callback.state_dict()["best_model_score"]
    except Exception as e:
        print(e)

    run.log(dict(score=score))
    wandb.join()
    sweep_q.put(WorkerDoneData(score=score))


def main():
    # Parse args
    cli = MyLightningCLI(
        trainer_class=TrainerWandb, 
        save_config_kwargs={
            'config_filename': 'config_pl.yaml',
            'overwrite': True,
        },
        args=[arg for arg in sys.argv[1:] if arg != 'fit'],
        run=False,
    )

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for fold_index in range(cli.config.data.init_args.k):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    # Run CV
    scores = {}
    for fold_index in range(cli.config.data.init_args.k):
        worker = workers[fold_index]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                fold_index=fold_index,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        scores[fold_index] = result.score

    sweep_run.log(dict(score=sum(scores.values()) / len(scores)))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    main()