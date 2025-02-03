import wandb
wandb.init(project="clm_no_traine")

import argparse, logging, os
import time
import shutil
import math
from scipy.stats import entropy
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
from models import GPT2LeastActionModel, GPT2IdLeastActionModel, GPT2DistanceLeastActionModel
from torch.utils.data import DataLoader
import scanpy as sc

from plots.plot_trajectories import map_embeddings_to_umap, plot

logger = get_logger(__name__)


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
    
    parser.add_argument("--alpha",
                        type=float,
                        default=0.9,
                        help="alpha")

    parser.add_argument("--dynamic_alpha",
                        # action="store_true",
                        action="store_false",
                        help="change alpha while training or not")


    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=12,
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
                        default=5,
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



def compute_shannon_entropy(generated_trajectories_ids, log_base=2):
    """
    Computes the Shannon entropy of visited cell indices across all generated trajectories.

    Parameters
    ----------
    generated_trajectories_ids : List of 1D numpy arrays
        Each array in the list corresponds to the sequence of cell indices visited by one trajectory.
    log_base : int or float, optional
        Base of the logarithm used in Shannon entropy. Common choices:
         - 2  for 'bits'
         - e  for natural log
         - 10 for 'log base 10'

    Returns
    -------
    float
        Shannon entropy of the distribution over visited states.
    """
    # 1) Flatten all generated cell indices
    all_cells = np.concatenate(generated_trajectories_ids, axis=0)  # shape: (total_steps_across_trajectories,)

    # 2) Count occurrences of each cell index.
    #    bin_count[i] = how many times 'i' was visited
    #    Note: i must be a non-negative integer index, which is true if these are adata.obs.index positions.
    counts = np.bincount(all_cells)

    # 3) Convert counts to probabilities
    total_visits = counts.sum()
    if total_visits == 0:
        return 0.0  # edge case: no visits at all
    probs = counts / total_visits  # each entry is fraction of total visits

    # 4) Compute Shannon entropy using scipy (or manually if you prefer).
    #    By default, `entropy` uses the natural log, you can specify `base=2` for bits, etc.
    shannon_ent = entropy(probs, base=log_base)

    return shannon_ent



def generate_sample_trajectories_grid(adata, model, epoch, top_p_values, top_k_values, temperature_values, output_dir):
    num_cells = len(adata)
    days_values = sorted(list(set(adata.obs["day_numerical"])))
    adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]

    n_trajectories = 200

    cell_types = list(set(adata.obs['cell_sets']))
    cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}

    for top_p in top_p_values:
        for top_k in top_k_values:
            for temperature in temperature_values:
                generated_trajectories_ids = []

                # Create organized folder structure
                result_dir = f"{output_dir}/epoch_{epoch}/top_p_{top_p}/top_k_{top_k}/temperature_{temperature}"
                os.makedirs(result_dir, exist_ok=True)

                print(f"Generating with top_p={top_p}, top_k={top_k}, temperature={temperature}")

                # Set generation configuration
                generation_config = GenerationConfig(
                    max_length=args.max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                )

                for _ in tqdm(range(n_trajectories)):
                    rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
                    cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
                    cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to('cuda:0')

                    # Generate text
                    output = model.generate(
                        input_ids=cell_idx.unsqueeze(0),
                        cell_type_ids=cell_type_idx.unsqueeze(0),
                        generation_config=generation_config,
                    )
                    generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])

                # Plot and save the result
                plot(
                    adata=adata,
                    sims=generated_trajectories_ids,
                    basis='X_draw_graph_fa',
                    cmap='rainbow',
                    linewidth=1.0,
                    linealpha=0.3,
                    dpi=300,
                    figsize=(12, 12),
                    ixs_legend_loc="upper right",
                    save=f"{result_dir}/trajectory_plot.png"
                )

                # Compute accuracy and coverage
                reference = adata.obs['day_numerical'].unique()
                mapped_trajectories_obs = []
                for trajectory in generated_trajectories_ids:
                    trajectory_obs = adata.obs['day_numerical'].iloc[trajectory].values
                    mapped_trajectories_obs.append(trajectory_obs)

                # calculate accuracy of predicted days
                matches = []
                for predicted_days in mapped_trajectories_obs:
                    if len(predicted_days) != len(reference):
                        continue
                    match_count = sum(1 for ref_day, pred_day in zip(reference, predicted_days) if ref_day == pred_day)
                    matches.append(match_count / len(predicted_days))

                accuracy = sum(matches) / len(matches)

                shannon_entropy = compute_shannon_entropy(generated_trajectories_ids, log_base=2)

                # --------------------
                # COVERAGE CALCULATION
                # --------------------
                # Convert generated trajectories to a NumPy array [num_trajectories x trajectory_length]

                real_trajectories_ids = np.array([np.array(traj) for traj in generated_trajectories_ids if len(traj)==args.max_length])
                days_values = sorted(list(set(adata.obs["day_numerical"])))
                n_trajectories = len(generated_trajectories_ids)

                unique_cell_id_per_day = [set(real_trajectories_ids[:, d]) for d in range(real_trajectories_ids.shape[1])]
                coverage_cells = [len(unique_cell_id_per_day[d]) / n_trajectories for d in range(len(days_values))]
                # Optionally take the mean coverage across days as a single metric
                coverage = np.mean(coverage_cells)

                # Save results to a file
                with open(f"{result_dir}/metrics.txt", "w") as f:
                    f.write(f"Accuracy: {accuracy:.4f}\n")
                    f.write(f"Shannon entropy: {shannon_entropy:.4f}\n")
                    f.write(f"Coverage (avg across days): {coverage:.4f}\n")

                # Log metrics to Weights & Biases
                wandb.log({
                    "epoch": epoch,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "accuracy": accuracy,
                    "shannon_entropy": shannon_entropy,
                    "coverage": coverage
                })

                print(f"Results saved in: {result_dir}")

    return accuracy, shannon_entropy


# def generate_sample_trajectories(adata, model, epoch, temperature=0.8):
#     num_cells = len(adata)
#     days_values = sorted(list(set(adata.obs["day_numerical"])))
#     adata_first_day = adata[adata.obs["day_numerical"] == days_values[0], :]
#
#     generated_trajectories_ids = []
#     temperature = temperature
#     top_k = 10
#     top_p = 0.3
#     n_trajectories = 100
#
#     generation_config = GenerationConfig(
#         max_length=args.max_length,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p,
#         do_sample=True,
#     )
#
#     cell_types = list(set(adata.obs['cell_sets']))
#     cell_types_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
#
#     for _ in tqdm(range(n_trajectories)):
#         rand_idx = np.random.choice(adata_first_day.obs.index, 1)[0]
#         cell_idx = torch.tensor([adata.obs.index.get_loc(rand_idx)], dtype=torch.long).to('cuda:0')
#         cell_type_idx = torch.tensor([cell_types_to_idx[adata.obs['cell_sets'][rand_idx]]], dtype=torch.long).to(
#             'cuda:0')
#         # Generate text
#         output = model.generate(
#             input_ids=cell_idx.unsqueeze(0),
#             cell_type_ids=cell_type_idx.unsqueeze(0),
#             generation_config=generation_config,
#         )
#         generated_trajectories_ids.append([x.cpu().numpy() for x in output.squeeze(0)])
#
#     plot(adata=adata,
#          sims=generated_trajectories_ids,
#          basis='X_draw_graph_fa',
#          cmap='rainbow',
#          linewidth=1.0,
#          linealpha=0.3,
#          dpi=300,
#          figsize=(12, 12),
#          ixs_legend_loc="upper right",
#          save=f"{args.output_dir}/epoch_{epoch}.png"
#          )
#     reference = adata.obs['day_numerical'].unique()
#     print('reference:', reference)
#     mapped_trajectories_obs = []
#     for trajectory in generated_trajectories_ids:
#         # Map each cell ID in the trajectory to its corresponding obs day information
#         trajectory_obs = adata.obs['day_numerical'].iloc[trajectory].values
#         mapped_trajectories_obs.append(trajectory_obs)
#     matches = []
#     for predicted_days in mapped_trajectories_obs:
#         if len(predicted_days) != len(reference):
#             # skip or handle
#             continue
#         # compare predicted day vs. the corresponding day in reference
#         match_count = sum(
#             1 for ref_day, pred_day in zip(reference, predicted_days) if ref_day == pred_day
#         )
#         matches.append(match_count / len(predicted_days))
#     accuracy = sum(matches) / len(matches)
#     coverage = len(np.unique(generated_trajectories_ids))/num_cells
#     return accuracy, coverage

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
        if os.path.isdir(os.path.join(output_dir, f)) and (f.startswith("step_"))
    ]
    checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))

    # Remove old checkpoints
    for checkpoint in checkpoints[:-n]:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        shutil.rmtree(checkpoint_path)
        print(f"Removed old checkpoint: {checkpoint_path}")

def linear_alpha_schedule(epoch, total_epochs, start_alpha, end_alpha):
    return start_alpha + (end_alpha - start_alpha) * (epoch / total_epochs)


if __name__ == "__main__":
    # set the random seed for reproducibility
    set_seed(42)
    args = parse_args()


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

    # model = GPT2IdLeastActionModel(config)
    model = GPT2DistanceLeastActionModel(config,
                                   cell_embeddings=torch.FloatTensor(adata.obsm["X_pca"]),
                                   alpha=args.alpha,
                                   )
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
        accelerator.init_trackers("clm_no_traine", experiment_config)

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

    evaluation_interval = 1000  # Evaluate every evaluation_interval steps

    top_p_values = [0.9]
    top_k_values = [1000]
    temperature_values = [0.8]


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if model.dist_loss_value is not None and model.ce_loss_value is not None:
                    wandb.log({
                        "dist_loss": model.dist_loss_value.item(),
                        "ce_loss": model.ce_loss_value.item()
                    }, step=completed_steps)

                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()

                if args.dynamic_alpha:
                    model.alpha = linear_alpha_schedule(epoch, total_epochs=args.num_train_epochs,
                                                        start_alpha=args.alpha, end_alpha=1)

                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                print(f"Saved checkpoint at step {completed_steps}")
                clean_old_checkpoints(args.output_dir, n=10)

            # Perform evaluation every `evaluation_interval` steps
            if completed_steps % evaluation_interval == 0:
                model.eval()
                losses = []
                for eval_step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    if model.dist_loss_value is not None and model.ce_loss_value is not None:
                        wandb.log({
                            "dist_loss": model.dist_loss_value.item(),
                            "ce_loss": model.ce_loss_value.item()
                        }, step=completed_steps)
                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")

                # accuracy, coverage = generate_sample_trajectories(adata, model, epoch)
                generate_sample_trajectories_grid(adata, model, epoch, top_p_values, top_k_values, temperature_values, args.output_dir)
                # logger.info(f"step {completed_steps}: accuracy: {accuracy} coverage: {coverage}")

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                model.train()  # Resume training mode after evaluation

            if completed_steps >= args.max_train_steps:
                break

        # model.eval()
        # losses = []
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     if model.dist_loss_value is not None and model.ce_loss_value is not None:
        #             wandb.log({
        #                 "dist_loss": model.dist_loss_value.item(),
        #                 "ce_loss": model.ce_loss_value.item()
        #             }, step=step)
        #     loss = outputs.loss
        #     losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        #
        # losses = torch.cat(losses)
        # try:
        #     eval_loss = torch.mean(losses)
        #     perplexity = math.exp(eval_loss)
        # except OverflowError:
        #     perplexity = float("inf")
        #
        # logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        #
        # # generate some sample trajectories
        # accuracy, coverage = generate_sample_trajectories(adata, model, epoch)
        # logger.info(f"epoch {epoch}: accuracy: {accuracy} coverage: {coverage}")
        #
        # if args.with_tracking:
        #     accelerator.log(
        #         {
        #             "perplexity": perplexity,
        #             "eval_loss": eval_loss,
        #             "train_loss": total_loss.item() / len(train_dataloader),
        #             "epoch": epoch,
        #             "step": completed_steps,
        #         },
        #         step=completed_steps,
        #     )
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            print(f"Saved checkpoint at epoch {epoch}")

            # Clean up old checkpoints
            # clean_old_checkpoints(args.output_dir, n=3)  # Retain the last 5 checkpoints
        wandb.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    # "accuracy": accuracy,
                    # "coverage": coverage,
                    "alpha":model.alpha
                },
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

    wandb.finish()
