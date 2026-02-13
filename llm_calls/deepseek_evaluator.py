import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import ValidationError

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from typing import Optional

# Support both package import (from llm_calls.prompts) and standalone import (from prompts)
try:
    from llm_calls.prompts import *
except ImportError:
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from prompts import *

class ResponseSchema(BaseModel):
    sentence_id_1: str # ids of the sentences being compared
    sentence_id_2: str # ids of the sentences being compared
    answers: str  # List of answers
    reasoning: str  # Reasoning behind the answers
    score: int  # Confidence score as an integer
    comment: str # Comment on confidence
    final_answer: str  # Final conclusion (YES/NO)

class EntailmentEvaluator:
    def __init__(self, model_name="deepseek-chat", prompt_type="default",
                api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url = base_url)
        self.model_name = model_name
        self.prompt_type = prompt_type
        
        # Select prompt template based on type

        self.prompt_template = _get_prompt_template(prompt_type)
        
        print(f"Using model: {model_name}")
        print(f"Using prompt type: {prompt_type}")

    def evaluate_pair_return_json_response(self, text1: str, text2: str, full_arg_1: Optional[str] = None, full_arg_2: Optional[str] = None, 
                                        context_1: Optional[str] = None, context_2: Optional[str] = None, 
                                        id1: Optional[int] = None, id2: Optional[int] = None):
        """
        Evaluate a pair of sentences in two API calls, one for each direction (text1 -> text2 and text2 -> text1).
        
        This method constructs two prompts using the provided arguments, sends the prompts to the OpenAI API, 
        and retrieves the responses for each direction. It returns a list of the two responses.
        
        Args:
            text1 (str): The first sentence to evaluate.
            text2 (str): The second sentence to evaluate.
            full_arg_1 (Optional[str]): An optional argument providing additional context for sentence 1.
            full_arg_2 (Optional[str]): An optional argument providing additional context for sentence 2.
            context_1 (Optional[str]): An optional context for sentence 1.
            context_2 (Optional[str]): An optional context for sentence 2.
            id1 (Optional[int]): The sentence ID for the first sentence.
            id2 (Optional[int]): The sentence ID for the second sentence.
        
        Returns:
            list: A list containing the responses for both directions of the evaluation.
        """
        # Format the prompt using the provided arguments
        prompt = self.prompt_template.format(text1=text1, text2=text2, argument_1=full_arg_1, argument_2=full_arg_2, context_1=context_1, context_2=context_2, sentence_id_1 = id1, sentence_id_2 = id2)
        
        # Get the response from both directions
        response = self._call_api_json(prompt)
 
        return response

    def _call_api_json(self, prompt: str):
        """
        Call the OpenAI API with the given prompt and return the response as a JSON object.
        
        This method sends a prompt to the OpenAI API, parses the response, and returns the parsed JSON object 
        if the response is valid. If parsing fails or an error occurs, it returns an error message.
        
        Args:
            prompt (str): The prompt to send to the OpenAI API.
        
        Returns:
            Union[dict, ResponseSchema]: The parsed JSON response, either as a dictionary or as a validated ResponseSchema object.
                                         If an error occurs, a dictionary with an error message is returned.
        """
        try:
            # deepseek-reasoner does not support response_format or temperature
            # It also needs a much higher max_tokens since reasoning tokens are separate
            max_tok = 8192 if "reasoner" in self.model_name else 1200
            api_kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tok,
            }
            if "reasoner" not in self.model_name:
                api_kwargs["response_format"] = {"type": "json_object"}
                api_kwargs["temperature"] = 0

            response = self.client.chat.completions.create(**api_kwargs)
            
            msg = response.choices[0].message
            content = msg.content or ''
            reasoning = getattr(msg, 'reasoning_content', '') or ''
            
            # Debug: log what we got back (first call only)
            if not hasattr(self, '_debug_logged'):
                print(f"[DEBUG] content length: {len(content)}, reasoning_content length: {len(reasoning)}")
                if content:
                    print(f"[DEBUG] content preview: {content[:300]}")
                if reasoning:
                    print(f"[DEBUG] reasoning preview: {reasoning[:300]}")
                self._debug_logged = True
            
            # Use content first; if empty, try to extract JSON from reasoning_content
            response_content = content if content.strip() else reasoning
            
            # Extract JSON from response (reasoner may wrap it in markdown code blocks)
            response_content = response_content.strip()
            if response_content.startswith("```"):
                # Strip markdown code fence
                lines = response_content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_content = "\n".join(lines).strip()
            
            # Try to find JSON object in the response
            if not response_content.startswith("{"):
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)
            
            try:
                response_dict = json.loads(response_content)
                
                try:
                    # Validate against Pydantic model
                    validated_response = ResponseSchema(**response_dict)
                    return validated_response
                except ValidationError as ve:
                    print(f"Validation error: {ve}")
                    return {"ERROR": f"Response validation failed: {str(ve)}"}
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {"ERROR": f"Invalid JSON format: {str(e)}"}  # Fixed error return
            
        except Exception as e:
            time.sleep(2)
            return {"ERROR": f"API call failed: {str(e)}"}

def batch_process_pairs(evaluator, data: pd.DataFrame, data_args: pd.DataFrame, path_intermediate: str, batch_size: int = 1, workers: int = 100):
    """
    Process multiple sentence pairs in parallel, collect the results, and export progress every N results.
    
    This version ensures that only the last two CSV files (current and previous) are kept to avoid excessive storage usage.
    
    Args:
        evaluator: The evaluator object that handles the evaluation of sentence pairs.
        data (pd.DataFrame): A DataFrame containing the sentence pairs to evaluate.
        data_args (pd.DataFrame): A DataFrame containing additional data related to the sentence pairs.
        batch_size (int): The number of results after which the progress is saved to a CSV file.
        path_intermediate (str): The base path to save intermediate results.
        workers (int): The number of parallel worker threads to use.
    
    Returns:
        List[Dict[str, Optional[float]]]: A list of dictionaries containing the evaluation results for each sentence pair.
    """
    results_separate = []  # List to store all results
    futures = []
    
    # Track the most recently saved intermediate file for cleanup
    last_saved_file = None

    # Set up thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:  # Adjust workers based on available resources
        for _, row in tqdm(data.iterrows(), total=len(data)):
            # Prepare the arguments for each row
            arg_id_1 = row['argument_id_1']
            arg_id_2 = row['argument_id_2']
            
            text1 = row['sentence_text_1']
            text2 = row['sentence_text_2']
            
            id1 = row['sentence_id_1']
            id2 = row['sentence_id_2']

            full_arg_1 = _get_external_argument_data(data_args, arg_id_1, 'argument', arg_id_col='argument_id')
            full_arg_2 = _get_external_argument_data(data_args, arg_id_2, 'argument', arg_id_col='argument_id')

            context_1 = _get_external_argument_data(data_args, arg_id_1, 'context', arg_id_col='argument_id')
            context_2 = _get_external_argument_data(data_args, arg_id_2, 'context', arg_id_col='argument_id')

            # Create a wrapper for the function that includes 'id1' and 'id2'
            futures.append(executor.submit(
                evaluator.evaluate_pair_return_json_response, 
                text1, text2, full_arg_1, full_arg_2, context_1, context_2, id1, id2))

        # Collect the results as they complete
        for count, future in enumerate(as_completed(futures), start=1):
            # Get the API response from the future result
            response = future.result()  # This should give us the tuple (response, row)
            
            # Prepare the result dictionary using the response and the sentence ids
            try:
                result = {
                    'sentence_id_1': response.sentence_id_1,
                    'sentence_id_2': response.sentence_id_2,
                    'answers_12': response.answers,
                    'reasonings_12': response.reasoning,
                    'comment_12': response.comment,
                    'llm_confidence_12': response.score,
                    'llm_conclusion_12': response.final_answer
                }
            except Exception as e:
                print(f'Exception {e} in: \n {response}')
                result = {
                    'sentence_id_1': None,
                    'sentence_id_2': None,
                    'answers_12': None,
                    'reasonings_12': None,
                    'comment_12': None,
                    'llm_confidence_12': None,
                    'llm_conclusion_12': None
                }

            # Append the result to the list
            results_separate.append(result)

            # Every N processes, export the results to a CSV file
            if count % batch_size == 0:
                print(f"Saving progress at batch {count}...")
                current_file = f"{path_intermediate}_progress_batch_{count}.csv"
                pd.DataFrame(results_separate).to_csv(current_file, index=False)
                
                # If there's already a saved file, delete the previous one
                if last_saved_file and os.path.exists(last_saved_file):
                    print(f"Deleting previous file: {last_saved_file}")
                    os.remove(last_saved_file)

                # Update the tracking variable
                last_saved_file = current_file

    return results_separate

def _load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from CSV, Excel, Text, or Pickle file based on the file extension.
    
    This function checks the file extension and loads the dataset accordingly. 
    It supports CSV, Excel (.xlsx), text (.txt), and Pickle (.pkl) formats.
    
    Args:
        file_path (str): The path to the dataset file (CSV, Excel, Text, or Pickle).
    
    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    file_extension = os.path.splitext(file_path)[1].lower()  # Extract the file extension and convert it to lowercase
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == '.txt':
        df = pd.read_csv(file_path, delimiter="\t")  # Assuming tab-separated for .txt files, adjust if necessary
    elif file_extension == '.pkl':
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return df

def _get_external_argument_data(df: pd.DataFrame, arg_id: str, data_column: str, arg_id_col: str = 'final_argument_id'):
    """
    Retrieve extra data from an argument such as context or full text.
    
    This function looks up data in a specified column of the DataFrame based on a given argument ID.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        arg_id (Union[int, str]): The ID of the argument to look up.
        data_column (str): The name of the column containing the data to retrieve.
        arg_id_col (str): The name of the column that contains the argument IDs (default: 'final_argument_id').
    
    Returns:
        Optional[str]: The data from the specified column for the given argument ID, or None if not found.
    """
    try:
        return df[df[arg_id_col] == arg_id][data_column].iloc[0]
    except IndexError:
        return None

def _get_prompt_template(prompt_type: str) -> str:
    """
    Helper function to get the corresponding prompt template based on prompt_type.
    
    This function selects the appropriate prompt template from predefined templates based on the input prompt type.
    
    Args:
        prompt_type (str): The type of prompt (e.g., 'default', 'test_prompt', etc.).
    
    Returns:
        str: The corresponding prompt template.
    """
    prompt_templates = {
        "default": DEFAULT_PROMPT_TEMPLATE,
        "test_prompt_tot_json2": TEST_PROMPT_TOT_JSON_BATCH,
        "deepseek_prompt_symmetric": PROMPT_TOT_DEEPSEEK,
        "deepseek_prompt_bb": PROMPT_TOT_DEEPSEEK_BB,
        "deepseek_prompt_bs": PROMPT_TOT_DEEPSEEK_BS,
        "deepseek_prompt_sb": PROMPT_TOT_DEEPSEEK_SB,
    }
    return prompt_templates.get(prompt_type, DEFAULT_PROMPT_TEMPLATE)

def save_sorted_subsets_by_column(df: pd.DataFrame, col1: str, col2: str, output_dir: str):
    """
    Save subsets of a DataFrame to individual .pkl files based on unique values in col1.
    Each subset is sorted by col2 and saved as '{unique_value}.pkl' in the specified output directory.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col1 : str
        Column name to group by.
    col2 : str
        Column name to sort by within each group.
    output_dir : str
        Directory where the .pkl files will be saved.

    Returns:
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for value in df[col1].dropna().unique():
        subset = df[df[col1] == value].sort_values(by=col2)
        filename = f"{value}.pkl"
        filepath = os.path.join(output_dir, filename)
        subset.to_pickle(filepath)
        print(len(subset))

def calculate_metrics(predictions: list, ground_truth: list):
    """
    Calculate evaluation metrics: F1 score, precision, recall, and accuracy.
    
    This function calculates common evaluation metrics for binary classification problems.
    
    Args:
        predictions (list): A list of predicted labels.
        ground_truth (list): A list of true labels.
    
    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics ('f1', 'precision', 'recall', 'accuracy').
    """
    f1 = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    accuracy = accuracy_score(ground_truth, predictions)
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }
    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Natural Language Inference')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', 
                        help='OpenAI model to use (default: gpt-4o-mini-2024-07-18)')
    parser.add_argument('--file', type=str, default='batch_1.xlsx',
                        help='Path to the Pickle file (default: batch_1.xlsx)')
    parser.add_argument('--external', type=str, default='13_arguments_no_ids.xlsx',
                        help='Path to the Pickle file (default: 13_arguments_no_ids.xlsx)')
    parser.add_argument('--prompt', type=str, 
                        choices=['default','test_prompt', 'test_prompt_tot', 'test_prompt_tot_json', 'test_prompt_tot_json2', 'deepseek_prompt_symmetric', 'deepseek_prompt_bb', 'deepseek_prompt_bs', 'deepseek_prompt_sb'],
                        default='default', 
                        help='Prompt type to use (default: default)')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output prefix for result files (default: based on prompt type)')
    args = parser.parse_args()

    # Set output prefix
    output_prefix = args.output if args.output else f"results_{args.prompt}"
    
    # Load data
    print(f"Loading data from {args.file}...")
    data = _load_data(args.file)
    print(f"Loading data from {args.external}...")
    data_args = _load_data(args.external)
    
    print(f"Loaded {len(data)} sentence pairs")
    
    # Initialize evaluator
    evaluator = EntailmentEvaluator(model_name=args.model, prompt_type=args.prompt)
    
    # Run batch processing
    print("Running batch evaluation...")
    results_separate = batch_process_pairs(evaluator, data, data_args, path_intermediate = args.output)
    
    # Save results to CSV
    separate_csv = f"{output_prefix}.csv"
    pd.DataFrame(results_separate).to_csv(separate_csv, index=False)
    print(f"Saved results to {separate_csv}")
