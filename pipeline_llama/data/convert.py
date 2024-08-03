import json

# Load the JSON data
with open('/home/devsrc/continue_pretrain/data/ruozhiba_qa_gpto.json', 'r') as f:
    data = json.load(f)

# Open a text file for writing
with open('ruozhiba_qa_gpto.txt', 'w', encoding='utf-8') as f:
    # Iterate over the JSON data
    for item in data:
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']

        # Write the instruction, input, and output to the text file
        f.write(f"Instruction: {instruction}\n")
        if input_text:
            f.write(f"Input: {input_text}\n")
        f.write(f"Output: {output}\n\n")