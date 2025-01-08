import argparse, logging, os
import time
import shutil
import math
from sre_parse import parse
from datasets import load_from_disk
import torch
import numpy as np
import logging
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GenerationConfig,
)
from tqdm import tqdm
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2LeastActionModel, GPT2IdLeastActionModel
from torch.utils.data import DataLoader
import scanpy as sc
import wandb
from plots.plot_trajectories import map_embeddings_to_umap, plot

logger = get_logger(__name__)

def compute_metrics(preds, labels):
    print("Computing metrics")
    return {}

def set_seed(seed: int) -> None:
    """Set random seed

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Model training configuration")

    parser.add_argument("--model_name",
        type=str,
        help="Path to the model")

    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1000000,
                        help="Number of epochs")

    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
                        )

    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=100,
                        help="Training batch size")


    parser.add_argument("--per_device_eval_batch_size",
                        type=int,
                        default=80,
                        help="Evaluation batch size")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate")

    parser.add_argument("--hidden_size",
                        type=int,
                        default=768,
                        help="Hidden size")

    parser.add_argument("--num_hidden_layers",
                        type=int,
                        default=6,
                        help="Number of hidden layers")

    parser.add_argument("--num_attention_heads",
                        type=int,
                        default=12,
                        help="Number of attention heads")

    parser.add_argument("--shard_size",
                        type=int,
                        default=10000,
                        help="Shard size")

    parser.add_argument("--train_data_path",
                        type=str,
                        help="Path to the training data")

    parser.add_argument("--eval_data_path",
                        type=str,
                        help="Path to the evaluation data")

    parser.add_argument("--output_dir",
                        type=str,
                        default="checkpoints/all_cells_vocabulary_no_trainer2",
                        help="Output directory")

    parser.add_argument("--max_length",
                        type=int,
                        default=39,
                        help="Maximum length")

    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Gradient accumulation steps")

    parser.add_argument("--dataloader_num_workers",
                        type=int,
                        default=1,
                        help="Number of workers for the dataloader.")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight decay to use.")

    parser.add_argument("--num_warmup_steps",
                        type=int,
                        default=0,
                        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument("--n_highly_variable_genes",
                        type=int,
                        default=2432,
                        help="Number of highly variable genes")

    parser.add_argument("--save_steps",
                        type=float,
                        default=0.01,
                        help="Save steps")

    parser.add_argument("--resume_from_checkpoint",
                        type=str,
                        default=None,
                        # default="/home/rohola/checkpoints/step_5000",
                        help="Resume from checkpoint")

    parser.add_argument("--with_tracking",
                        action="store_false",
                        help="Whether to enable experiment trackers for logging.")

    parser.add_argument("--generate_trajectories",
                        action="store_true",
                        help="Generate trajectories"
                        )

    parser.add_argument("--report_to",
                        type=str,
                        default="wandb",
                        help="Report to")

    parser.add_argument("--project_dir",
                        type=str,
                        default=".",
                        help="Project directory")

    parser.add_argument("--lr_scheduler_type",
                        type=SchedulerType,
                        default=SchedulerType.LINEAR,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],)

    parser.add_argument("--checkpointing_steps",
                        type=str,
                        default=100,
                        help="Checkpointing steps")

    parser.add_argument("--expression_max_value",
                        type=float,
                        default=10.0,
                        help="Expression max value")

    parser.add_argument("--expression_min_value",
                        type=float,
                        default=0.0,
                        help="Expression min value")

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device")

    return parser.parse_args()

def generate_sample_trajectories(adata, model, epoch):
    num_cells = len(adata)
    days_values = sorted(list(set(adata.obs["day_numerical"])))
    adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

    generated_trajectories_ids = []
    temperature = 0.8
    top_k = 10
    top_p = 0.3
    n_trajectories = 100

    generation_config = GenerationConfig(
        max_length=args.max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    cell_types = list(set(adata.obs['cell_sets']))
    cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}

    for _ in tqdm(range(n_trajectories)):
        rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
        cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
        cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to(
            'cuda:0')
        # Generate text
        output = model.generate(
            input_ids=cell_idx.unsqueeze(0),
            cell_type_ids=cell_type_idx.unsqueeze(0),
            generation_config=generation_config,
        )
        generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])

    plot(adata=adata,
         sims=generated_trajectories_ids,
         basis='X_draw_graph_fa',
         cmap='rainbow',
         linewidth=1.0,
         linealpha=0.3,
         dpi=300,
         figsize=(12, 12),
         ixs_legend_loc="upper right",
         save=f"{args.output_dir}/epoch_{epoch}.png"
         )
    
    mapped_trajectories_obs = []
    reference = adata.obs['day_numerical'].unique() # already ordered
    for trajectory in generated_trajectories_ids:
        # Map each cell ID in the trajectory to its corresponding obs day information
        trajectory_obs = adata.obs['day_numerical'].iloc[trajectory].values 
        mapped_trajectories_obs.append(trajectory_obs)
    matches = []
    for lst in mapped_trajectories_obs:
        # Count the matches and get an % of correctness
        match_count = sum(1 for ref, val in zip(reference, lst) if ref == val)
        matches.append(match_count/len(lst))
    accuracy = sum(matches)/len(matches)
    coverage = len(np.unique(generated_trajectories_ids))/num_cells
    return accuracy, coverage 

def custom_collate_fn(batch):
    """
    Custom collate function to modify 'cell_type_ids' in the batch.
    """
    for item in batch:
        if 'cell_type_ids' in item:
            item['cell_type_ids'] = None
    return default_data_collator(batch)


def clean_old_checkpoints(output_dir, n):
    """
    Retain only the last `n` checkpoints and remove older ones.

    Args:
        output_dir (str): Directory where checkpoints are saved.
        n (int): Number of recent checkpoints to retain.
    """
    checkpoints = [
        f for f in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, f)) and (f.startswith("step_") or f.startswith("epoch_"))
    ]
    checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))

    # Remove old checkpoints
    for checkpoint in checkpoints[:-n]:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        shutil.rmtree(checkpoint_path)
        print(f"Removed old checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # set the random seed for reproducibility
    set_seed(42)

    args = parse_args()


    os.environ["WANDB_PROJECT"] = "BioLeastAction2"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)


    # adata = sc.read_h5ad("data/reprogramming_schiebinger_force_directed_768.h5ad")
    adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

    # assert adata.obsm["X_pca"].shape[1] == args.hidden_size, f"PCA dimension {adata.obsm['X_pca'].shape[1]} is not equal to hidden size {args.hidden_size}"

    if args.generate_trajectories:
        train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                                  adata=adata,
                                                  columns_to_use=["input_ids", "labels", "cell_type_ids"],
                                                  T=0.8,
                                                  embedding_size=adata.obsm["X_pca"].shape[1],
                                                  shuffle=True)
    else:
        dataset = load_from_disk('data/adata_trajectory_dataset_hf')
        train_dataset = dataset['train']
        eval_dataset = dataset['test']


    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        # collate_fn=default_data_collator,
        collate_fn=custom_collate_fn, # use custom collate_fn to modify 'cell_type_ids'
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        # collate_fn=default_data_collator,
        collate_fn=custom_collate_fn, # use custom collate_fn to modify 'cell_type_ids'
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers
    )

    num_cells = len(adata)
    # num_cell_types = len(set(adata.obs['cell_sets']))

    config = GPT2Config(
        n_positions=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
        vocab_size=num_cells, #+ num_cell_types, # number of cells and cell types
        use_cache=False,
    )

    model = GPT2IdLeastActionModel(config)
    model.to(args.device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None:
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    print(f"Saved checkpoint at step {completed_steps}")

                    # Clean up old checkpoints
                    clean_old_checkpoints(args.output_dir, n=5)  # Retain the last 5 checkpoints

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        # generate some sample trajectories
        accuracy, coverage = generate_sample_trajectories(adata, model, epoch)
        logger.info(f"epoch {epoch}: accuracy: {accuracy} coverage: {coverage}")


        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    "accuracy":accuracy,
                    "coverage":coverage
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            print(f"Saved checkpoint at epoch {epoch}")

            # Clean up old checkpoints
            clean_old_checkpoints(args.output_dir, n=3)  # Retain the last 5 checkpoints

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

    wandb.finish()
