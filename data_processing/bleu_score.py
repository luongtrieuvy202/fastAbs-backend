import os
import json
import evaluate




import argparse
from nltk.translate.bleu_score import sentence_bleu

def remove_punctuation(input_string):
    punctuation = ['.', ',', '"']
    cleaned_string = input_string
    for char in punctuation:
        cleaned_string = cleaned_string.replace(char, '')
    return cleaned_string


def calculate_f1_score(dec_dir, test_dir):
    count = 0
    scorer = evaluate.load('bleu')
    score = 0
    for filename in os.listdir(dec_dir):
        if filename.endswith(".dec"):
            json_file = filename.split('.')[0] + '.json'
            dec_filepath = os.path.join(dec_dir, filename)
            paragraph = ''
            count = count + 1
            #print(count)
            # print(json_file)
            # print(filename)
            try:
                with open(dec_filepath, 'r') as dec_file:
                    for line in dec_file:
                        line = line.strip()
                        if not (line.endswith('.') or line.endswith(',')):
                            line += '.'
                        paragraph += ' ' + line
            except FileNotFoundError:
                print(f"DEC file {dec_filepath} not found.")
            except Exception as e:
                print(f"Error loading DEC file {dec_filepath}: {str(e)}")

            json_filepath = os.path.join(test_dir, json_file)

            reference_sum = ''

            with open(json_filepath, 'r') as json_file:
                data = json.load(json_file)
                abstract_value = data.get('abstract', None)
                reference_sum = ' '.join(abstract_value)
            paragraph = remove_punctuation(paragraph).replace('_',' ').lower().strip()
            reference_sum = remove_punctuation(reference_sum).replace('_',' ').lower().strip()
            if not paragraph:
                count -= 1
                continue
            score += scorer.compute(max_order=2,smooth=True, references=[[reference_sum]], predictions=[paragraph])['bleu']
      
            
    print(count)
    return score/count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate F1 score for text data.")
    parser.add_argument("--dec_dir", type=str, help="Directory containing DEC files")
    parser.add_argument("--test_dir", type=str, help="Directory containing JSON files for testing")

    args = parser.parse_args()

    if not args.dec_dir or not args.test_dir:
        print("Both --dec_dir and --test_dir are required.")
    else:
        result = calculate_f1_score(args.dec_dir, args.test_dir)
        print("Average BLEU Score:", result)