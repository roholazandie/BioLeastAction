from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig

# Load the checkpoint
checkpoint_path = "path_to_your_checkpoint"  # specify the checkpoint path here
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to('cuda:0')

generation_config = GenerationConfig(
    max_length=100,  # adjust the max_length as needed
    num_return_sequences=1,  # number of sequences to generate
    do_sample=True,
    top_p=0.9,
    top_k=50,
    early_stopping=True
)


# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate new tokens
input_text = "You are"  # starting text for generation
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda:0')

# Generate text
output = model.generate(
    input_ids,
    generation_config=generation_config
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
