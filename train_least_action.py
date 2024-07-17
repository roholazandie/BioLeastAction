import argparse, logging, os
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
from data_utils.cell_differentiation_datasets import get_dataset
from models import GPT2LeastActionModel
import wandb

# Enable logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # Enable logging for transformers
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description="Model training configuration")

    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=9, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=80, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
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
    args = parse_args()

    os.environ["WANDB_PROJECT"] = "BioLeastAction"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ['WANDB_NAME'] = 'reprogramming_schiebinger'

    train_dataset = get_dataset(dataset_name="reprogramming_schiebinger", embedding_size=args.hidden_size)
    eval_dataset = get_dataset(dataset_name="reprogramming_schiebinger", embedding_size=args.hidden_size, shuffle=False)

    config = GPT2Config(
        n_positions=args.max_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers,
        n_head=args.num_attention_heads,
    )

    model = GPT2LeastActionModel(config)
    model.to(args.device)

    working_dir = f"{args.output_dir}"

    training_args = TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=40,
        # gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=1e-10,
        warmup_ratio=0.1,
        logging_dir=working_dir,
        dataloader_num_workers=20,
        logging_steps=10,
        save_strategy="steps",  # save a checkpoint every save_steps
        save_steps=int(args.save_steps * len(train_dataset)),
        save_total_limit=5,
        eval_strategy="steps",  # evaluation is done every eval_steps
        eval_steps=int(0.25 * len(train_dataset)),
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
        # data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    # Train the model
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    wandb.finish()