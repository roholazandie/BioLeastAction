import argparse, logging, os
import torch
import numpy as np
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2LeastActionModel, GPT2IdLeastActionModel, GPT2CellLeastActionModel
import scanpy as sc
import wandb


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

    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--n_epochs", type=int, default=1000000, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=9, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=80, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--shard_size", type=int, default=10000, help="Shard size")
    parser.add_argument("--train_data_path", type=str, help="Path to the training data")
    parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--max_length", type=int, default=39, help="Maximum length")
    parser.add_argument("--n_highly_variable_genes", type=int, default=2432, help="Number of highly variable genes")
    parser.add_argument("--save_steps", type=float, default=0.01, help="Save steps")
    parser.add_argument("--expression_max_value", type=float, default=10.0, help="Expression max value")
    parser.add_argument("--expression_min_value", type=float, default=0.0, help="Expression min value")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()


if __name__ == "__main__":
    # set the random seed for reproducibility
    set_seed(42)

    args = parse_args()

    os.environ["WANDB_PROJECT"] = "BioLeastActionCell"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_768.h5ad")

    train_dataset, eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger",
                                              adata=adata,
                                              embedding_size=args.hidden_size,
                                              shuffle=True)

    num_cells = len(adata)
    num_cell_types = len(set(adata.obs['cell_sets']))

    assert args.hidden_size < adata.shape[1], "The hidden size must be less than the number of genes"
    assert args.hidden_size == adata.obsm['X_pca'].shape[1], "The hidden size must be equal to the embedding size"

    config = GPT2Config(
        n_positions=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
        vocab_size=num_cells,
        cell_type_vocab_size=num_cell_types,
    )

    model = GPT2CellLeastActionModel(config)
    # model = GPT2CellLeastActionModel.from_pretrained("checkpoints/all_cells_vocabulary_cell_type/checkpoint-15000")
    model.to(args.device)

    working_dir = f"{args.output_dir}/all_cells_vocabulary_cell_type_cell_embedding"

    training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=50, #350
        per_device_eval_batch_size=50, # 300
        # gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        # weight_decay=1e-10,
        # max_grad_norm=0.1,
        # warmup_ratio=0.1,
        # lr_scheduler_type="cosine",
        logging_dir=working_dir,
        dataloader_num_workers=20,
        logging_steps=10,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=500,  #int(args.save_steps * len(train_dataset)),
        save_total_limit=5,
        eval_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=2000, #int(0.25 * len(train_dataset)),
        eval_accumulation_steps=1,
        load_best_model_at_end=False,
        fp16=True,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # optimizers=(optimizer, scheduler)
    )

    # Train the model
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)

    # Evaluate the model
    trainer.evaluate()

    wandb.finish()
