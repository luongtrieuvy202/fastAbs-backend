import os
import json
import jsonlines

file_path = 'output.txt'  # Replace 'your_file.txt' with the actual path to your text file

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read all lines from the file and store them in a list
    lines = file.readlines()




filename = "example_requests_to_parallel_process.jsonl"
n_requests = len(lines)

with jsonlines.open(filename, mode='w') as writer:
    for i in range(n_requests):
        job = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant in in classification category in Vietnamese."},
                {"role": "user", "content": " 10 Bài báo sau thuộc chủ đề nào trong bốn chủ đề sau thế_giới , pháp_luật , tin_tức , kinh_tế . Câu trả lời chỉ được bao gồm tên chủ đề cho mỗi bài báo: " + lines[i]}
            ],
        }
        writer.write(job)
        print('Processing articles number:', i)
