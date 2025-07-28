from unsloth import FastLanguageModel
import torch
import os

models_dir = os.environ.get("MODELS_DIR")
if models_dir is None:
    raise EnvironmentError("MODELS_DIR environment variable is not set. Please set it to the directory where your models are stored.")

datasets_dir = os.environ.get("DATASETS_DIR")
if datasets_dir is None:
    raise EnvironmentError("DATASETS_DIR environment variable is not set. Please set it to the directory where your datasets are stored.")

model_name = "DeepSeek-R1-Distill-Llama-8B"
model_path = os.path.join(models_dir, model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist. Please ensure the model is downloaded and available.")

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    # fast_inference = True, # Enable vLLM fast inference
    # max_lora_rank = lora_rank,
    full_finetuning = False, # We have full finetuning now!
    # gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

import re
from datasets import load_dataset, Dataset

reasoning_dataset = load_dataset("/workspace/datasets/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("/workspace/datasets/FineTome-100k", split = "train")
print(f"reasoning_dataset:\n{reasoning_dataset}")
print(f"non_reasoning_dataset:\n{non_reasoning_dataset}")

def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }

reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_dataset.map(generate_conversation, batched = True)["conversations"],
    tokenize = False,
)

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(non_reasoning_dataset)

non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)

print(f"number of original reasoning conversations: {len(reasoning_conversations)}")
print(f"number of original non reasoning conversations: {len(non_reasoning_conversations)}")
print("Let's select 75% reasoning and 25% chat based")
chat_percentage = 0.25
import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 2407,
)
print(f"number of reasoning conversations: {len(reasoning_conversations)}")
print(f"number of non reasoning conversation subset: {len(non_reasoning_subset)}")
print(f"ratio of non_reasoning in overall dataset: {len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations))}")

data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

max_prompt_length = 256

prompt_text = "Solve (x + 2)^2 = 0."
# prompt_text = "Calculate pi."
# prompt_text = "How many r's are in strawberry?"

print(f"** Before Lora trining **")
print(f"== disable thinking ==")
messages = [
    {"role" : "user", "content" : prompt_text}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

print(f"== enable thinking ==")
messages = [
    {"role" : "user", "content" : prompt_text}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # enable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024, # Increase for longer outputs!
    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print(f"** After Lora trining **")
print(f"== disable thinking ==")
messages = [
    {"role" : "user", "content" : prompt_text}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

print(f"== enable thinking ==")
messages = [
    {"role" : "user", "content" : prompt_text}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = True, # enable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024, # Increase for longer outputs!
    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

model.save_pretrained(model_path+"_lora_model")  # Local saving
tokenizer.save_pretrained(model_path+"_lora_model")
model.save_pretrained_merged(model_path+"_lora_merged_16bit", tokenizer, save_method = "merged_16bit",)
model.save_pretrained_gguf(model_path+"_lora_f16", tokenizer, quantization_method = "f16")
model.save_pretrained_gguf(model_path+"_lora_q4_k_m", tokenizer, quantization_method = "q4_k_m")
