import os
import json
import time
import re
import csv # Import the csv module
from datetime import datetime # For timestamp
from dotenv import load_dotenv
from pathlib import Path # Import Path for robust file handling
from typing import Set # Import Set for type hinting

# --- LlamaIndex specific imports ---
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
# Corrected Google GenAI import based on user feedback and package structure
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.file import PDFReader

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()
print("Attempting to load environment variables...")

# Configure API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Setup Logging ---
LOG_FILENAME = f"framework_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_FIELDNAMES = [
    'Timestamp', 'Iteration', 'A1_Model', 'A2_Model', 'Question',
    'A1_Internal_Approach', 'A1_Expected_Answer', 'A2_Provided_Answer',
    'A2_Provided_Reasoning', 'Evaluation_Result', 'Failure_Analysis'
]

# Function to initialize log file and write header if needed
def initialize_log_file(filename, fieldnames):
    file_exists = os.path.isfile(filename)
    # Open in append mode, create if doesn't exist
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()
            print(f"Initialized log file: {filename}")

# Function to write a log entry
def write_log_entry(filename, fieldnames, log_data):
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Ensure all potential keys exist in log_data, provide defaults if not
        entry = {field: log_data.get(field, '') for field in fieldnames}
        writer.writerow(entry)
# --- End Logging Setup ---


if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    print("Please ensure it's set in your environment or a .env file.")
    exit()
else:
    print("Google API Key found.")


# If A2 is always Google, you could remove this check.
if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not found.")
    print("This may not be an issue if you are not using any OpenAI models.")
else:
    print("OpenAI API Key found.")


# Define the models to use via LlamaIndex
A1_MODEL_ID = "models/gemini-2.5-pro-preview-05-06" 

A2_MODEL_ID = "models/gemini-2.0-flash" 


PDF_FILENAME = "aptiv.pdf" # <<< RENAME this to your PDF filename if needed

# Generation configuration (applied during LLM instantiation)
TEMPERATUREA1 = 0.5 
TEMPERATUREA2 = 0.3 

# --- Helper Function for Robust JSON Extraction (No changes needed) ---
def extract_json_from_response(text: str) -> dict | None:
    """
    Extracts a JSON object from a string, handling potential markdown code blocks.
    (Handles ```json ... ``` or just { ... })
    """
    # Regex to find JSON block
    match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", text, re.DOTALL | re.MULTILINE)
    if match:
        json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse extracted JSON. Error: {e}\nExtracted String:\n---\n{json_str}\n---")
            return None
    else:
        # Fallback: Try parsing the whole text if no explicit block found
        try:
            cleaned_text = text.strip()
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                print("Warning: No JSON markers found, attempting to parse entire response as JSON.")
                return json.loads(cleaned_text)
            else:
                print(f"Error: Could not find JSON block and response doesn't appear to be raw JSON.\nResponse Text:\n---\n{text}\n---")
                return None

        except json.JSONDecodeError:
            print(f"Error: Could not find JSON block or parse the full response.\nResponse Text:\n---\n{text}\n---")
            return None

# --- LLM API Call Function (No changes needed) ---
def call_llm(llm_instance: LLM, prompt: str, model_name_for_log: str) -> str | None:
    """
    Calls the specified LlamaIndex LLM instance with the given prompt.
    """
    print(f"\n--- Calling LlamaIndex LLM ({model_name_for_log}) ---")
    # print(f"Prompt:\n{prompt}\n--- End Prompt ---") # Uncomment for debugging
    try:
        response = llm_instance.complete(prompt)
        response_text = response.text
        print(f"--- LLM Response (Raw) ---")
        # print(response_text) # Uncomment for debugging raw output
        print("--- End Raw Response ---")
        return response_text
    except Exception as e:
        print(f"Error calling LlamaIndex LLM {model_name_for_log}: {e}")
        if "API key" in str(e):
            print("Please double-check the relevant API key (Google or OpenAI).")
        return None


# --- Agent Functions (MODIFIED generate_question_internal_solution) ---


def generate_question_internal_solution(llm_a1: LLM, excerpt_text: str, previous_questions: Set[str]) -> dict | None:
    """
    Agent A1: Generates question, internal approach, and answer using its LLM,
    avoiding previously generated questions.
    """
    avoid_list_str = ""
    if previous_questions:
        avoid_list_str = "\n\nIMPORTANT: Avoid generating questions similar to these previous ones:\n"
        avoid_list_str += "\n".join(f"- \"{q}\"" for q in previous_questions)

    prompt = f"""
You are an expert financial analyst (Agent A1).
Based *only* on the provided financial document excerpt:
\"\"\"{excerpt_text}\"\"\"
{avoid_list_str}

Task:
1.  Generate *one* ***NEW***, challenging, and tricky numerical question that requires calculation or synthesis of information found *strictly within the excerpt*.
    * The question must be substantially different from any previous questions listed above.
    * The question must be directly answerable by calculation or synthesis using *only* the numerical data and statements present in the excerpt.
    * It should go beyond simply looking up a single number and test understanding of financial concepts or relationships between numbers presented.
    * ***CRITICAL CONSTRAINT: DO NOT generate hypothetical or assumption-based questions.*** Avoid questions starting with phrases like "Assuming...", "If...", "What if...", or any similar conditional clauses that require information or scenarios not explicitly stated in the text. The question must ask for a concrete value derivable *only* from the provided text.
    * Avoid questions that require external knowledge or data not present in the text.
2.  Internally determine the precise step-by-step approach required to calculate the answer using *only* the provided text. Clearly reference the specific numbers or data points from the text used in each step.
3.  Calculate the expected numerical answer based *only* on your derived approach and the provided text. Ensure the final answer format is a simple number or percentage where appropriate (e.g., "0.33", "25%", "150.5").

Output Format:
Respond *only* with a single, valid JSON object containing the keys 'question', 'internal_approach', and 'expected_answer'. Do not include any introductory text, explanations, comments, or markdown formatting outside the JSON object itself.

Example JSON Output (ensure your question is different if this example was listed above):
{{
  "question": "Calculate the company's Debt-to-Equity ratio for the year 2023 based on the provided balance sheet.",
  "internal_approach": "1. Find Total Liabilities for 2023 ($1050 million) in the Balance Sheet. 2. Find Total stockholders' equity for 2023 ($1450 million) in the Balance Sheet. 3. Divide Total Liabilities by Total Equity (1050 / 1450).",
  "expected_answer": "0.724"
}}
"""
    response_text = call_llm(llm_a1, prompt, A1_MODEL_ID)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    # Optional: Add a check here to see if the generated question is *still* in previous_questions
    # If it is, you might return None to force a retry in the next iteration.
    # if data and data.get('question') in previous_questions:
    #     print("Warning: A1 generated a duplicate question despite instructions. Skipping this attempt.")
    #     return None

    if data and 'question' in data and 'internal_approach' in data and 'expected_answer' in data:
        print("A1 Generation Successful (JSON Parsed).")
        return data
    else:
        print(f"Error: A1 generation response parsing failed or missing required keys.")
        return None

# (attempt_solution_independent function remains unchanged)
def attempt_solution_independent(llm_a2: LLM, excerpt_text: str, question: str) -> dict | None:
    """
    Agent A2: Attempts to answer the question independently using its LLM.
    """
    prompt = f"""
You are an analyst agent (Agent A2). You need to answer a question based *only* on the provided financial document excerpt. You must figure out the method yourself using only the information given.

Given the following financial document excerpt:
\"\"\"{excerpt_text}\"\"\"

Answer the following question *strictly based on the excerpt provided*:
"{question}"

Instructions:
1.  Carefully analyze the question and the excerpt.
2.  Devise your own step-by-step method to find the relevant information *only within the excerpt* and perform any necessary calculations. Clearly state the data points you are using from the text.
3.  Provide your final numerical answer. Ensure the final answer format is a simple number or percentage where appropriate (e.g., "0.5", "50%", "210.2").
4.  Clearly explain the reasoning and the exact steps you took to arrive at your answer, referencing data found *only in the excerpt*.

Output Format:
Respond *only* with a single, valid JSON object containing the keys 'answer' and 'reasoning'. Do not include any introductory text, explanations, comments, or markdown formatting outside the JSON object itself.

Example JSON Output:
{{
  "answer": "0.72",
  "reasoning": "1. Identified Total Liabilities for 2023 as $1050 million from the Balance Sheet section in the provided text. 2. Identified Total stockholders' equity for 2023 as $1450 million from the Balance Sheet section. 3. Calculated Debt-to-Equity ratio as Total Liabilities / Total Equity = $1050 / $1450 = 0.7241. Rounded to 0.72."
}}
"""
    response_text = call_llm(llm_a2, prompt, A2_MODEL_ID)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'answer' in data and 'reasoning' in data:
        print("A2 Solution Attempt Successful (JSON Parsed).")
        return data
    else:
        print(f"Error: A2 solving response parsing failed or missing required keys.")
        return None

# (evaluate_independent_solution function remains unchanged)
def evaluate_independent_solution(llm_a1: LLM, question_data: dict, a2_response: dict) -> dict | None:
    """
    Agent A1: Evaluates A2's independent solution attempt using its LLM.
    """
    # Helper function to attempt converting answers to floats for comparison
    def _to_float(value_str):
        try:
            # Remove common currency symbols, commas, and percentage signs for conversion
            cleaned_str = str(value_str).replace('$', '').replace(',', '').replace('%', '').strip()
            return float(cleaned_str)
        except (ValueError, TypeError):
            return None

    a1_answer_float = _to_float(question_data.get('expected_answer'))
    a2_answer_float = _to_float(a2_response.get('answer'))
    numerical_match_flag = "Could not compare numerically"
    if a1_answer_float is not None and a2_answer_float is not None:
        # Allow for small tolerance in floating point comparison
        # Increased tolerance slightly for broader matching
        is_close = abs(a1_answer_float - a2_answer_float) < 0.02 * abs(a1_answer_float) or abs(a1_answer_float - a2_answer_float) < 0.2
        numerical_match_flag = "Yes" if is_close else "No"
    elif str(question_data.get('expected_answer', '')).strip() == str(a2_response.get('answer', '')).strip():
        # Fallback to string comparison if floats failed but strings match exactly
        numerical_match_flag = "Yes (Exact String Match)"


    prompt = f"""
You are the expert financial analyst (Agent A1) who originally created a question and determined the correct approach/answer based *only* on a specific financial document excerpt. You are now evaluating an attempt by another agent (Agent A2) who had to devise their *own* method to solve the question using *only* the same excerpt.

Your Original Question:
"{question_data.get('question', 'N/A')}"

Your Internal Step-by-Step Approach (for calculating the expected answer from the excerpt):
"{question_data.get('internal_approach', 'N/A')}"

Your Calculated Expected Answer (derived strictly from the excerpt):
"{question_data.get('expected_answer', 'N/A')}"
---
Agent A2's Attempt:

A2's Provided Answer:
"{a2_response.get('answer', 'N/A')}"

A2's Provided Reasoning/Method:
"{a2_response.get('reasoning', 'N/A')}"
---
Evaluation Task:
Perform the following evaluation based *only* on the information provided above and the assumption that both agents were restricted to the *same original excerpt*.

1.  **Numerical Correctness:** Compare A2's Provided Answer to Your Calculated Expected Answer.
    * My Expected Answer: {question_data.get('expected_answer', 'N/A')}
    * A2's Answer: {a2_response.get('answer', 'N/A')}
    * Are they numerically equivalent, allowing for minor rounding differences (e.g., 0.724 vs 0.72)? (Pre-calculated Numerical Match: {numerical_match_flag})

2.  **Method Validity & Reasoning:** Analyze A2's Provided Reasoning/Method.
    * Does A2's reasoning demonstrate understanding of the question's core financial concept?
    * Did A2 devise a *logically valid* method to arrive at *an* answer using data potentially found in the original excerpt? (Focus on logical flow, not whether it matches your exact method).
    * Does A2's reasoning clearly state which data points it *claims* to have used from the excerpt?
    * Does the calculation described in A2's reasoning logically lead to A2's provided answer? (Check A2's internal consistency).
    * Critically: Is A2's method and reasoning plausible *given the constraint that it must rely solely on the original excerpt*? Does it seem to invent data or use external knowledge?

3.  **Overall Result:** Classify the attempt as 'Success' or 'Failure'.
    * 'Success' requires BOTH the numerical answer to be essentially correct (as per Numerical Match Flag starting with 'Yes') AND the reasoning/method (A2's Reasoning/Method) to be sound, logically valid, and demonstrably based *only* on information plausibly available in the original excerpt.
    * 'Failure' occurs if the numerical answer is incorrect (Numerical Match Flag = 'No') OR if A2's reasoning/method is flawed (e.g., uses the wrong financial formula, misunderstands terms, references data likely outside the excerpt, makes significant calculation errors within its own stated logic, or the reasoning doesn't support the answer provided).

4.  **Failure Analysis (If Applicable):** If the result is 'Failure', provide a *brief* and specific 'analysis' explaining the primary reason(s) for the failure based on the criteria in step 3. Focus on *why* it failed according to the rules.

Output Format:
Respond *only* with a single, valid JSON object.
- Include the key 'result' with a value of either 'Success' or 'Failure'.
- ONLY if the 'result' is 'Failure', ALSO include the key 'analysis' with your brief explanation. Do not include any introductory text, explanations, comments, or markdown formatting outside the JSON object itself.

Example Success JSON Output:
{{
  "result": "Success"
}}

Example Failure JSON Output:
{{
  "result": "Failure",
  "analysis": "A2 correctly identified the need for Total Liabilities and Equity but used the 2022 figures ($850/$1350) instead of the required 2023 figures ($1050/$1450) from the excerpt, leading to an incorrect numerical answer."
}}
"""
    response_text = call_llm(llm_a1, prompt, f"{A1_MODEL_ID} (Evaluation)")
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'result' in data:
        if data['result'] == 'Failure' and 'analysis' not in data:
            print(f"Warning: A1 evaluation response is 'Failure' but missing 'analysis' key. Adding placeholder.")
            data['analysis'] = "Evaluation marked Failure, but A1 did not provide specific analysis."
        elif data['result'] not in ['Success', 'Failure']:
            print(f"Error: A1 evaluation response 'result' key has invalid value: {data['result']}")
            return None
        print(f"A1 Evaluation Successful (JSON Parsed). Result: {data['result']}")
        return data
    else:
        print(f"Error: A1 evaluation response parsing failed or missing 'result' key.")
        return None


# --- Orchestrator Function  ---
def run_framework(pdf_path: Path, llm_a1: LLM, llm_a2: LLM, target_failures: int = 2, max_iterations: int = 5) -> list:
    """
    Orchestrates the interaction between A1 and A2 using LlamaIndex LLMs,
    logging all iterations to CSV and collecting failures, avoiding repeated questions.
    """
    failure_count = 0
    failure_log_list = [] # Still collect failures for final JSON report/summary
    iteration = 0
    previous_questions = set() # <--- Initialize set to store previous questions

    # --- Load PDF Content ---
    # (PDF Loading code remains the same)
    pdf_file_str = str(pdf_path) # Convert Path to string for printing/exists check
    print(f"\nLoading financial excerpt from: {pdf_file_str}")
    if not pdf_path.is_file():
        print(f"Error: PDF file not found at {pdf_file_str}")
        return []
    try:
        reader = PDFReader()
        documents = reader.load_data(file=pdf_path) # Pass Path object
        if not documents:
            print(f"Error: No content could be extracted from {pdf_file_str}")
            return []
        excerpt_text = "\n\n".join([doc.text for doc in documents])
        print(f"Successfully loaded and extracted text from PDF (length: {len(excerpt_text)} chars).")
        if len(excerpt_text) < 100:
            print("Warning: Extracted text seems very short. Ensure the PDF contains sufficient data.")
    except Exception as e:
        print(f"Error reading or parsing PDF file {pdf_file_str}: {e}")
        return []
    # --- PDF Loading End ---

    print(f"\nStarting framework run. Target Failures: {target_failures}, Max Iterations: {max_iterations}")
    print(f"Using A1 Model: {A1_MODEL_ID}, A2 Model: {A2_MODEL_ID}")
    print(f"Logging iterations to: {LOG_FILENAME}")
    initialize_log_file(LOG_FILENAME, LOG_FIELDNAMES) # Ensure header is written

    while failure_count < target_failures and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*15} Iteration {iteration} {'='*15}")
        time.sleep(2) # Rate limiting delay

        q_data = None
        a2_resp = None
        eval_result = None
        current_log_data = { # Initialize log data for this iteration
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Iteration': iteration,
            'A1_Model': A1_MODEL_ID,
            'A2_Model': A2_MODEL_ID,
            'Question': 'GENERATION FAILED',
            'A1_Internal_Approach': '', 'A1_Expected_Answer': '',
            'A2_Provided_Answer': 'GENERATION FAILED', 'A2_Provided_Reasoning': '',
            'Evaluation_Result': 'N/A', 'Failure_Analysis': 'A1 Generation Failed'
        }


        # 1. A1 generates question, passing previous questions
        print("\nStep 1: A1 generating question and internal solution...")
        # Pass the set of previous questions to the generator function
        q_data = generate_question_internal_solution(llm_a1, excerpt_text, previous_questions)

        if q_data:
            generated_question = q_data.get('question', 'N/A')
            print(f"A1 Generated Question: {generated_question}")
            # Add the newly generated question to the set for next iteration
            previous_questions.add(generated_question) # <--- Add question to the set
            current_log_data.update({
                'Question': generated_question,
                'A1_Internal_Approach': q_data.get('internal_approach', 'N/A'),
                'A1_Expected_Answer': q_data.get('expected_answer', 'N/A'),
                'A2_Provided_Answer': 'ATTEMPT SKIPPED', # Default if A2 fails
                'Failure_Analysis': '' # Clear initial failure reason
            })
        else:
            print(f"ERROR: Iteration {iteration} - A1 generation failed.")
            write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
            continue


        # 2. A2 attempts solution (remains the same)
        if q_data:
            print("\nStep 2: A2 attempting independent solution...")
            a2_resp = attempt_solution_independent(llm_a2, excerpt_text, q_data['question'])
            if a2_resp:
                print(f"A2 Answer: {a2_resp.get('answer', 'N/A')}")
                print(f"A2 Reasoning:\n{a2_resp.get('reasoning', 'N/A')}")
                current_log_data.update({
                    'A2_Provided_Answer': a2_resp.get('answer', 'N/A'),
                    'A2_Provided_Reasoning': a2_resp.get('reasoning', 'N/A'),
                    'Evaluation_Result': 'ATTEMPT SKIPPED', # Default if evaluation fails
                    'Failure_Analysis': ''
                })
            else:
                print(f"ERROR: Iteration {iteration} - A2 solving failed.")
                current_log_data.update({
                    'A2_Provided_Answer': 'SOLVING FAILED',
                    'A2_Provided_Reasoning': '',
                    'Evaluation_Result': 'N/A',
                    'Failure_Analysis': 'A2 Solving Failed'
                })
                write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
                continue


        # 3. A1 evaluates solution (remains the same)
        if q_data and a2_resp:
            print("\nStep 3: A1 evaluating A2's independent solution...")
            eval_result = evaluate_independent_solution(llm_a1, q_data, a2_resp)
            if eval_result:
                print(f"Evaluation Result: {eval_result.get('result', 'N/A')}")
                current_log_data.update({
                    'Evaluation_Result': eval_result.get('result', 'N/A'),
                    'Failure_Analysis': eval_result.get('analysis', '')
                })
            else:
                print(f"ERROR: Iteration {iteration} - A1 evaluation failed.")
                current_log_data.update({
                    'Evaluation_Result': 'EVALUATION FAILED',
                    'Failure_Analysis': 'A1 Evaluation Failed'
                })
                write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
                continue


        # 4. Log results to CSV & handle failure counting (remains the same)
        write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)

        if eval_result and eval_result.get('result') == 'Failure':
            failure_count += 1
            analysis = eval_result.get('analysis', 'No analysis provided by evaluator.')
            print(f"***** FAILURE LOGGED ({failure_count}/{target_failures}) *****")
            print(f"Failure Analysis: {analysis}")
            failure_log_list.append({
                "iteration": iteration,
                "question": q_data.get('question', 'N/A'),
                "a1_internal_approach": q_data.get('internal_approach', 'N/A'),
                "a1_expected_answer": q_data.get('expected_answer', 'N/A'),
                "a2_provided_answer": a2_resp.get('answer', 'N/A'),
                "a2_provided_reasoning": a2_resp.get('reasoning', 'N/A'),
                "failure_analysis": analysis,
                "a1_model": A1_MODEL_ID,
                "a2_model": A2_MODEL_ID
            })
        elif eval_result and eval_result.get('result') == 'Success':
            print("----- SUCCESS ----- (A2's independent approach was valid)")
        elif not eval_result and (q_data and a2_resp):
            print("----- ITERATION ENDED (Evaluation Error) -----")
        elif not a2_resp and q_data:
            print("----- ITERATION ENDED (A2 Solving Error) -----")
        elif not q_data:
            print("----- ITERATION ENDED (A1 Generation Error) -----")


        print(f"{'='*40}") # Separator for iterations

    # --- Loop End ---
    # (Loop end logic remains the same)
    if iteration >= max_iterations and failure_count < target_failures:
        print(f"\nWarning: Reached max iterations ({max_iterations}) before reaching target failures ({target_failures}).")
    elif failure_count >= target_failures:
        print(f"\nTarget number of failures ({target_failures}) reached.")

    print(f"\nFramework run finished. Detailed log available in: {LOG_FILENAME}")
    return failure_log_list


# --- Main Execution Block ---
# (No changes needed in __main__ block regarding this modification)
if __name__ == "__main__":
    # Basic check if keys were loaded
    if not google_api_key: # Only Google API key is essential now if both models are Google
        print("Google API Key not configured properly. Exiting.")

    else:
        print("\n" + "="*50)
        print(" Starting Multi-Agent Framework Run with LlamaIndex ")
        print("="*50)

        # --- Instantiate LLMs ---
        try:
            print(f"Instantiating A1 LLM: {A1_MODEL_ID}")
            llm_a1 = GoogleGenAI(
                model_name=A1_MODEL_ID,
                api_key=google_api_key,
                temperature=TEMPERATUREA1
            )
            print(f"Instantiating A2 LLM: {A2_MODEL_ID}")
            llm_a2 = GoogleGenAI( # Changed from OpenAI
                model_name=A2_MODEL_ID, # Use the new A2_MODEL_ID
                api_key=google_api_key, # Use Google API Key
                temperature=TEMPERATUREA2
            )
            # MODIFICATION END
            print("LLM instances created successfully.")
        except Exception as e:
            print(f"FATAL: Failed to instantiate LLMs: {e}")
            if "Model not found" in str(e) or "Could not find model" in str(e):
                # This message now applies to both A1_MODEL_ID and A2_MODEL_ID
                print(f"Please verify that the model IDs ('{A1_MODEL_ID}', '{A2_MODEL_ID}') are correct and accessible with your API key.")
            exit()
        # --- LLM Instantiation End ---

        TARGET_FAILURES = 2
        MAX_ITERATIONS = 5

        pdf_file_path = Path(PDF_FILENAME)

        if not pdf_file_path.is_file():
            print(f"\nFATAL ERROR: PDF file '{PDF_FILENAME}' not found in the current directory.")
            print("Please make sure the file exists and the filename is correct in the script.")
        else:
            failure_report = run_framework(
                pdf_path=pdf_file_path,
                llm_a1=llm_a1,
                llm_a2=llm_a2,
                target_failures=TARGET_FAILURES,
                max_iterations=MAX_ITERATIONS
            )

            print("\n" + "="*50)
            print(f" Failure Summary Report (Logged {len(failure_report)} failures) ")
            print(f" Complete iteration log in: {LOG_FILENAME} ")
            print("="*50)

            if not failure_report:
                print("No failures meeting the criteria were logged within the iteration limit.")
            else:
                report_filename = f"failure_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        json.dump(failure_report, f, indent=4, ensure_ascii=False)
                    print(f"Failure summary saved to: {report_filename}")
                except Exception as e:
                    print(f"Error saving failure summary to JSON: {e}")

                for i, failure in enumerate(failure_report):
                    print(f"\n--- Failure Summary #{i+1} (Iteration: {failure['iteration']}) ---")
                    print(f"  Models: A1={failure['a1_model']}, A2={failure['a2_model']}")
                    print(f"  Question: {failure['question']}")
                    print(f"  Failure Analysis: {failure['failure_analysis']}")


    print("\n" + "="*50)
    print(" Framework Simulation Complete ")
    print("="*50)