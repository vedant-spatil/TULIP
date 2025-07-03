def main():
    from datasets import load_dataset
    from colorama import Fore
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, prepare_model_for_kbit_training

    import os
    from dotenv import load_dotenv
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    load_dotenv()

    torch.cuda.empty_cache()

    hf_token = os.getenv("HF_API_KEY")

    dataset = load_dataset("data", split="train")
    print(Fore.YELLOW + str(dataset[0]) + Fore.RESET)

    def format_chat_template(batch, tokenizer):
        system_prompt =  """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            
        samples = []
        questions = batch["question"]
        answers = batch["answer"]
        for i in range (len(questions)):
            row_json = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": answers[i]}
            ]
            text = tokenizer.apply_chat_template(row_json, tokenize=False)
            samples.append(text)
        return {
            "instruction": questions,
            "response": answers,
            "text": samples
        }

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code = True,
        token = hf_token,
    )

    train_dataset = dataset.map(lambda x:format_chat_template(x,tokenizer), num_proc=8, batched=True, batch_size=10)

    print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        device_map = "auto",
        token = hf_token,
        cache_dir = "./workspace"
    )

    print(Fore.CYAN + str(model) + Fore.RESET)
    print(Fore.LIGHTYELLOW_EX + str(next(model.parameters()).device))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(output_dir="TinyLlama/TinyLlama-1.1B-Chat-v1.0-SFT", 
                       num_train_epochs=5, 
                       per_device_train_batch_size=1, 
                       gradient_accumulation_steps=4, 
                       bf16=False, #Set True Comment if using Cuda  
                       fp16=False, #Set True if using T4 and Cuda
                       report_to=[]),
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("complete_checkpoint")
    trainer.model.save_pretrained("final_model")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()