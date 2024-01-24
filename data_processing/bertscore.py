from bert_score import BERTScorer
import os
import json


scorer = BERTScorer(lang="vn",model_type="microsoft/deberta-xlarge-mnli")





import argparse

def calculate_f1_score(dec_dir, test_dir):
    count = 0
    score = 0
    scorer = BERTScorer(lang="vn")

    for filename in os.listdir(dec_dir):
        if filename.endswith(".dec"):
            json_file = filename.split('.')[0] + '.json'
            dec_filepath = os.path.join(dec_dir, filename)
            paragraph = ''
            count = count + 1
            print(count)
            try:
                with open(dec_filepath, 'r') as dec_file:
                    for line in dec_file:
                        line = line.strip()
                        if not line.endswith('.') and not line.endswith(','):
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

            P1, R1, F1_1 = scorer.score([paragraph], [reference_sum])
            score += F1_1

    return score / count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate F1 score for text data.")
    parser.add_argument("--dec_dir", type=str, help="Directory containing DEC files")
    parser.add_argument("--test_dir", type=str, help="Directory containing JSON files for testing")

    args = parser.parse_args()

    if not args.dec_dir or not args.test_dir:
        print("Both --dec_dir and --test_dir are required.")
    else:
        result = calculate_f1_score(args.dec_dir, args.test_dir)
        print("Average F1 Score:", result)