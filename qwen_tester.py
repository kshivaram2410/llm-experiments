from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import pandas as pd
from datetime import datetime

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("No GPU available")

def get_gpu_memory_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
    return 0

def build_prompt():
    """
    """
    product_info = """
    The Sony WH-1000XM5 wireless headphones feature industry-leading noise cancellation technology, 
    offering up to 30 hours of battery life on a single charge. They come with a premium price tag of $399.99 
    and are available in black and silver colors. The headphones weigh 250 grams and include a 3.5mm audio jack 
    for wired listening. They support Bluetooth 5.2 and are compatible with both iOS and Android devices. 
    The ear cups are made of soft memory foam and the headband is adjustable for maximum comfort.
    """

    prompt = f"""
    You are a expert information extraction agent and are tasked with extracting information from the following product description.
    
    Product description:
    {product_info}

    Required attributes to extract:
    - brand
    - model
    - price
    - colors
    - battery_life
    - weight
    - connectivity_type
    - compatibility
    - materials
    - noise_cancellation
    
    Please format your response as a valid JSON object with these exact attribute names.
    """

    return prompt

models = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

def run_inference(model_name, num_runs=4):
    print(f"\n{'='*50}")
    print(f"Running inference with {model_name}")
    print(f"{'='*50}")
    
    print("\nInitial GPU memory state:")
    print_gpu_memory()

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )

    print(f"\nModel loaded on {model.device}")

    print("\nGPU memory after model loading:")
    print_gpu_memory()

    # prepare the model input
    prompt = build_prompt()
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGPU memory after preparing inputs:")
    print_gpu_memory()

    # Store latencies
    latencies = []
    final_output = None

    # Run inference multiple times
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # measure inference time
        start_time = time.time()
        
        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        inference_time = time.time() - start_time
        latencies.append(inference_time)
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        print(f"Inference time: {inference_time:.2f} seconds")

        # Only store the output from the last run
        if run == num_runs - 1:
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            final_output = {
                'thinking_content': thinking_content,
                'content': content
            }

    # Calculate statistics
    cold_start_latency = latencies[0]
    warm_start_latencies = latencies[1:]
    avg_warm_start_latency = sum(warm_start_latencies) / len(warm_start_latencies)
    
    print(f"\nLatency Statistics:")
    print(f"Cold start latency: {cold_start_latency:.2f} seconds")
    print(f"Average warm start latency: {avg_warm_start_latency:.2f} seconds")

    # Get final GPU memory usage before cleanup
    final_gpu_memory = get_gpu_memory_used()

    # Clean up GPU memory
    del model
    del tokenizer
    del model_inputs
    del generated_ids
    torch.cuda.empty_cache()
    
    print("\nGPU memory after cleanup:")
    print_gpu_memory()

    return {
        'model_name': model_name,
        'gpu_memory_used_gb': final_gpu_memory,
        'cold_start_latency': cold_start_latency,
        'avg_warm_start_latency': avg_warm_start_latency,
        'input_prompt': prompt,
        'thinking_content': final_output['thinking_content'],
        'predicted_output': final_output['content']
    }

def save_results_to_csv(results, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    return df

# Run inference for all models and collect results
results = []
for model_name in models:
    result = run_inference(model_name, num_runs=4)
    results.append(result)

# Save results to CSV and display the comparison table
df = save_results_to_csv(results)
print("\nComparison Table:")
print(df[['model_name', 'gpu_memory_used_gb', 'cold_start_latency', 'avg_warm_start_latency']].to_string())
