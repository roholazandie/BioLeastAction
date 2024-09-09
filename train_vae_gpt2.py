import argparse, logging, os
from transformers import GPT2Config, TrainingArguments, Trainer
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2VAEModel
import wandb
from train_least_action_id import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Model training configuration")

    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--n_epochs", type=int, default=100000, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=9, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=80, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n_gene", type=int, default=19089, help="Hidden size")
    parser.add_argument("--layer_size1", type=int, default=2048, help="Hidden size")
    # parser.add_argument("--layer_size2", type=int, default=2048, help="Hidden size")
    # parser.add_argument("--layer_size3", type=int, default=64, help="Hidden size")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads")
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
    set_seed(42)

    args = parse_args()

    os.environ["WANDB_PROJECT"] = "BioLeastAction"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ['WANDB_NAME'] = 'tree_vectors'

    dataset_name = "tree_vectors"
    branch_factors = [3, 4, 2]
    steps = 100
    args.n_gene = 500
    train_dataset, eval_dataset = get_dataset(dataset_name=dataset_name,
                                            branching_factors=branch_factors,
                                                steps=steps,
                                              embedding_size=args.n_gene,
                                              shuffle=False)

    args.max_length = steps * len(branch_factors) + 1

    config = GPT2Config(
        n_positions=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
        n_gene=args.n_gene
    )

    model = GPT2VAEModel(config)
    model.to(args.device)

    working_dir = f"{args.output_dir}/tree_vectors_{':'.join([str(x) for x in branch_factors])}_{steps}"

    training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=100000,
        per_device_eval_batch_size=100000,
        # gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        # weight_decay=1e-10,
        # warmup_ratio=0.1,
        logging_dir=working_dir,
        dataloader_num_workers=20,
        logging_steps=10,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=int(args.save_steps * len(train_dataset)),
        save_total_limit=5,
        eval_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=int(len(train_dataset)),
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
    )

    # Train the model
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    wandb.finish()