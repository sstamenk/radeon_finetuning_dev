from unsloth import FastLanguageModel
import torch
import os
import argparse
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Unsloth and GRPO")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-Llama-3.1-8B-Instruct",
        help="Name of the model to fine-tune (default: meta-Llama-3.1-8B-Instruct)"
    )
    return parser.parse_args()

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(datasets_dir: str, split = "train") -> Dataset:
    gsm8k_path= os.path.join(datasets_dir, "gsm8k")
    data = load_dataset(gsm8k_path, 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def main():
    args = parse_arguments()

    # Check environment variables
    models_dir = os.environ.get("MODELS_DIR")
    if models_dir is None:
        raise EnvironmentError("MODELS_DIR environment variable is not set. Please set it to the directory where your models are stored.")

    datasets_dir = os.environ.get("DATASETS_DIR")
    if datasets_dir is None:
        raise EnvironmentError("DATASETS_DIR environment variable is not set. Please set it to the directory where your datasets are stored.")

    # Set up model parameters
    model_name = args.model_name
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist. Please ensure the model is downloaded and available.")

    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower

    # Load and configure model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = False,  # False for LoRA 16bit
        fast_inference = True,  # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.9,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",  # Enable long context finetuning
        random_state = 3407,
    )

    # Load dataset
    dataset = get_gsm8k_questions(datasets_dir)

    # Set up training configuration
    max_prompt_length = 256

    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,  # Increase to 4 for smoother training
        num_generations = 6,  # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        # num_train_epochs = 1,  # Set to 1 for a full training run
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "none",  # Can use Weights & Biases
        output_dir = "outputs",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    # Train the model
    trainer.train()

    # Test the model before and after training
    prompt_text = "Calculate pi."
    # prompt_text = "How many r's are in strawberry?"
    text = tokenizer.apply_chat_template([
        {"role" : "user", "content" : prompt_text},
    ], tokenize = False, add_generation_prompt = True)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    
    # Test before training
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    print(f"** Before GRPO training **\n{output}")

    # Save LoRA adapter
    model.save_lora("grpo_saved_lora")

    # Test after training
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    print(f"** After GRPO training **\n{output}")

    # Save final models
    model.save_pretrained_merged(model_path+"_GRPO", tokenizer, save_method = "merged_16bit",)
    model.save_pretrained_gguf(model_path+"_GRPO_f16", tokenizer, quantization_method = "f16")
    model.save_pretrained_gguf(model_path+"_GRPO_q4_k_m", tokenizer, quantization_method = "q4_k_m")


if __name__ == "__main__":
    main()
