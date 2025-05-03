import google.generativeai as genai
import os
import json
import time
import re # For robust JSON extraction

# --- Configuration ---

# Configure the Generative AI SDK
try:
    # Attempt to get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("Google AI SDK Configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    print("Please ensure you have installed the SDK (`pip install google-generativeai`)")
    print("and set the GOOGLE_API_KEY environment variable.")
    exit() # Exit if configuration fails

# Define the models to use (Using Flash for cost-effectiveness, Pro can also be used)
# Consider using a more powerful model like 'gemini-1.5-pro-latest' for A1 if needed
A1_MODEL_ID = "gemini-1.5-flash-latest" # "Superior" LLM (can be Pro)
A2_MODEL_ID = "gemini-1.5-flash-latest" # "Inferior" LLM (can be Flash)

# Safety settings for the model (adjust if needed)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Generation configuration (optional, adjust temperature for creativity/determinism)
GENERATION_CONFIG = {
    "temperature": 0.3, # Lower temperature for more factual/predictable output
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 8192, # Adjust as needed
    "response_mime_type": "text/plain", # Ensure plain text for easier parsing
}


# --- Helper Function for Robust JSON Extraction ---

def extract_json_from_response(text: str) -> dict | None:
    """
    Extracts a JSON object from a string, handling potential markdown code blocks.

    Args:
        text: The raw text response from the LLM.

    Returns:
        A dictionary if JSON is found and parsed, otherwise None.
    """
    # Regex to find JSON block within ```json ... ``` or just { ... }
    match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", text, re.DOTALL | re.MULTILINE)
    if match:
        json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse extracted JSON. Error: {e}\nExtracted String: {json_str}")
            return None
    else:
        # Fallback: Try parsing the whole text if no explicit block found
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Error: Could not find JSON block or parse the full response.\nResponse Text: {text}")
            return None

# --- LLM API Call Function ---

def call_gemini_model(model_id: str, prompt: str) -> str | None:
    """
    Calls the specified Gemini model with the given prompt.

    Args:
        model_id: The ID of the Gemini model to use (e.g., "gemini-1.5-flash-latest").
        prompt: The input prompt string.

    Returns:
        The text response from the model, or None if an error occurs.
    """
    print(f"\n--- Calling Gemini Model ({model_id}) ---")
    # print(f"Prompt:\n{prompt}\n--- End Prompt ---") # Uncomment for debugging
    try:
        model = genai.GenerativeModel(
            model_id,
            safety_settings=SAFETY_SETTINGS,
            generation_config=GENERATION_CONFIG
            )
        response = model.generate_content(prompt)
        print(f"--- Gemini Response (Raw) ---")
        # print(response.text) # Uncomment for debugging raw output
        print("--- End Raw Response ---")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini model {model_id}: {e}")
        # Handle specific API errors if needed (e.g., rate limits, authentication)
        if "API key not valid" in str(e):
            print("Please check your GOOGLE_API_KEY.")
        return None

# --- Agent Functions ---

def generate_question_internal_solution(gemini_model_id: str, excerpt: str) -> dict | None:
    """
    Agent A1: Generates a question, determines the internal solution approach,
    and calculates the expected answer based on the excerpt using Gemini.

    Args:
        gemini_model_id: The identifier for the Gemini model to use.
        excerpt: The 10-K excerpt text.

    Returns:
        A dictionary with 'question', 'internal_approach', 'expected_answer'
        keys, or None if an error occurs.
    """
    prompt = f"""
You are an expert financial analyst (Agent A1).
Based *only* on the provided 10-K excerpt:
\"\"\"{excerpt}\"\"\"

Task:
1.  Generate *one* challenging numerical question that requires calculation or synthesis of information found within the excerpt. The question should go beyond simply looking up a single number and test understanding of financial concepts or relationships between numbers presented.
2.  Internally determine the precise step-by-step approach required to calculate the answer using *only* the provided text.
3.  Calculate the expected numerical answer based on your derived approach.

Output Format:
Respond *only* with a single JSON object containing the keys 'question', 'internal_approach', and 'expected_answer'. Do not include any introductory text, explanations, or markdown formatting outside the JSON object itself.

Example JSON Output:
{{
  "question": "Calculate the company's Debt-to-Equity ratio for the year 2023.",
  "internal_approach": "1. Find Total Liabilities for 2023 ($500). 2. Find Total Equity for 2023 ($1500). 3. Divide Total Liabilities by Total Equity (500 / 1500).",
  "expected_answer": "0.33"
}}
"""
    response_text = call_gemini_model(gemini_model_id, prompt)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'question' in data and 'internal_approach' in data and 'expected_answer' in data:
        print("A1 Generation Successful (JSON Parsed).")
        return data
    else:
        print(f"Error: A1 generation response parsing failed or missing required keys.")
        return None


def attempt_solution_independent(gemini_model_id: str, excerpt: str, question: str) -> dict | None:
    """
    Agent A2: Attempts to answer the question by devising its own method based
    on the excerpt using Gemini.

    Args:
        gemini_model_id: The identifier for the Gemini model to use.
        excerpt: The 10-K excerpt text.
        question: The question generated by A1.

    Returns:
        A dictionary with 'answer' and 'reasoning' keys, or None if an error occurs.
    """
    prompt = f"""
You are an analyst agent (Agent A2). You need to answer a question based *only* on the provided 10-K excerpt. You must figure out the method yourself.

Given the following 10-K excerpt:
\"\"\"{excerpt}\"\"\"

Answer the following question:
"{question}"

Instructions:
1.  Carefully analyze the question and the excerpt.
2.  Devise your own step-by-step method to find the relevant information in the excerpt and perform any necessary calculations. Use *only* the information present in the excerpt.
3.  Provide your final numerical answer.
4.  Clearly explain the reasoning and the exact steps you took to arrive at your answer.

Output Format:
Respond *only* with a single JSON object containing the keys 'answer' and 'reasoning'. Do not include any introductory text, explanations, or markdown formatting outside the JSON object itself.

Example JSON Output:
{{
  "answer": "0.5",
  "reasoning": "1. Identified Total Liabilities for 2023 as $500 from the Balance Sheet. 2. Identified Total Equity for 2023 as $1000 from the Balance Sheet. 3. Calculated Debt-to-Equity ratio as Total Liabilities / Total Equity = $500 / $1000 = 0.5."
}}
"""
    response_text = call_gemini_model(gemini_model_id, prompt)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'answer' in data and 'reasoning' in data:
         print("A2 Solution Attempt Successful (JSON Parsed).")
         return data
    else:
        print(f"Error: A2 solving response parsing failed or missing required keys.")
        return None


def evaluate_independent_solution(gemini_model_id: str, question_data: dict, a2_response: dict) -> dict | None:
    """
    Agent A1: Evaluates A2's independent solution attempt using Gemini.

    Args:
        gemini_model_id: The identifier for the Gemini model to use.
        question_data: The dictionary returned by generate_question_internal_solution.
                       Expected keys: 'question', 'internal_approach', 'expected_answer'.
        a2_response: The dictionary returned by attempt_solution_independent.
                     Expected keys: 'answer', 'reasoning'.

    Returns:
        A dictionary with 'result' ('Success' or 'Failure') and optionally
        'analysis' keys, or None if an error occurs.
    """
    prompt = f"""
You are the expert financial analyst (Agent A1) who originally created a question and determined the correct approach/answer. You are now evaluating an attempt by another agent (Agent A2) who had to devise their *own* method to solve the question using only the provided 10-K excerpt.

Your Original Question:
"{question_data.get('question', 'N/A')}"

Your Internal Step-by-Step Approach (for calculating the expected answer):
"{question_data.get('internal_approach', 'N/A')}"

Your Calculated Expected Answer:
"{question_data.get('expected_answer', 'N/A')}"
---
Agent A2's Attempt:

A2's Provided Answer:
"{a2_response.get('answer', 'N/A')}"

A2's Provided Reasoning/Method:
"{a2_response.get('reasoning', 'N/A')}"
---
Evaluation Task:
Perform the following evaluation based *only* on the information provided above:

1.  **Numerical Correctness:** Compare A2's Provided Answer to Your Calculated Expected Answer. Is A2's answer numerically correct? Allow for minor rounding differences (e.g., 41.67% vs 41.7%).
2.  **Method Validity & Reasoning:** Analyze A2's Provided Reasoning/Method.
    *   Did A2 seem to understand the core concept of the question?
    *   Did A2 devise a *valid* method (even if different from yours) using *only* data present in the (implied) excerpt?
    *   Did A2 correctly identify the necessary data points according to *its chosen method*?
    *   Did A2 perform the calculations correctly according to *its chosen method*?
3.  **Overall Result:** Classify the attempt as 'Success' or 'Failure'.
    *   'Success' requires BOTH the numerical answer to be correct AND the reasoning/method to be sound and logically valid based on the excerpt.
    *   'Failure' occurs if the numerical answer is incorrect OR if A2's reasoning/method demonstrates a significant flaw (e.g., used the wrong financial formula, misinterpreted terms, selected incorrect data from the excerpt, made major calculation errors within its own stated logic).
4.  **Failure Analysis (If Applicable):** If the result is 'Failure', provide a *brief* and specific 'analysis' explaining the primary reason for the failure (e.g., "Calculation error in step 2 of A2's reasoning.", "A2 used Gross Profit instead of Operating Income for Operating Margin calculation.", "Incorrectly identified 'Total Assets' instead of 'Total Equity'.").

Output Format:
Respond *only* with a single JSON object.
- Include the key 'result' with a value of either 'Success' or 'Failure'.
- ONLY if the 'result' is 'Failure', ALSO include the key 'analysis' with your brief explanation.

Example Success JSON Output:
{{
  "result": "Success"
}}

Example Failure JSON Output:
{{
  "result": "Failure",
  "analysis": "A2 used the 2022 Revenue figure instead of 2023 in the final calculation."
}}
"""
    response_text = call_gemini_model(gemini_model_id, prompt)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'result' in data:
        if data['result'] == 'Failure' and 'analysis' not in data:
             print(f"Error: A1 evaluation response is 'Failure' but missing 'analysis' key.")
             return None # Or potentially return the data but flag the issue
        elif data['result'] not in ['Success', 'Failure']:
             print(f"Error: A1 evaluation response 'result' key has invalid value: {data['result']}")
             return None
        print(f"A1 Evaluation Successful (JSON Parsed). Result: {data['result']}")
        return data
    else:
        print(f"Error: A1 evaluation response parsing failed or missing 'result' key.")
        return None

# --- Orchestrator Function ---

def run_framework(excerpt: str, target_failures: int = 2, max_iterations: int = 5) -> list:
    """
    Orchestrates the interaction between A1 and A2 using Gemini models, logging failures.

    Args:
        excerpt: The 10-K excerpt text.
        target_failures: The number of failures to log before stopping.
        max_iterations: Maximum number of cycles to run to prevent infinite loops
                        and manage API costs/usage.

    Returns:
        A list of dictionaries, where each dictionary details a failure event.
    """
    failure_count = 0
    failure_log = []
    iteration = 0

    print(f"Starting framework run. Target Failures: {target_failures}, Max Iterations: {max_iterations}")
    print(f"Using A1 Model: {A1_MODEL_ID}, A2 Model: {A2_MODEL_ID}")

    while failure_count < target_failures and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*15} Iteration {iteration} {'='*15}")

        # Simple delay to potentially avoid hitting rate limits if running fast
        time.sleep(1)

        # 1. A1 generates question, internal approach, and expected answer
        print("\nStep 1: A1 generating question and internal solution...")
        q_data = generate_question_internal_solution(A1_MODEL_ID, excerpt)
        if not q_data:
            print(f"ERROR: Iteration {iteration} skipped due to failure in A1 generation.")
            continue # Skip to next iteration

        print(f"A1 Generated Question: {q_data.get('question', 'N/A')}")
        # print(f"A1 Internal Approach: {q_data.get('internal_approach', 'N/A')}") # Debug
        # print(f"A1 Expected Answer: {q_data.get('expected_answer', 'N/A')}")   # Debug

        # 2. Orchestrator passes excerpt and question ONLY to A2
        print("\nStep 2: A2 attempting independent solution...")
        a2_resp = attempt_solution_independent(A2_MODEL_ID, excerpt, q_data['question'])
        if not a2_resp:
             print(f"ERROR: Iteration {iteration} skipped due to failure in A2 solving.")
             continue # Skip to next iteration

        print(f"A2 Answer: {a2_resp.get('answer', 'N/A')}")
        print(f"A2 Reasoning:\n{a2_resp.get('reasoning', 'N/A')}")

        # 3. Orchestrator passes A1's internal data and A2's response back to A1 for evaluation
        print("\nStep 3: A1 evaluating A2's independent solution...")
        eval_result = evaluate_independent_solution(A1_MODEL_ID, q_data, a2_resp)
        if not eval_result:
            print(f"ERROR: Iteration {iteration} evaluation could not be completed due to failure in A1 evaluation.")
            # Optionally log this as a system failure or just skip
            continue # Skip logging this iteration's result

        print(f"Evaluation Result: {eval_result.get('result', 'N/A')}")

        # 4. Check result and log failures
        if eval_result.get('result') == 'Failure':
            failure_count += 1
            analysis = eval_result.get('analysis', 'No analysis provided by evaluator.')
            print(f"***** FAILURE LOGGED ({failure_count}/{target_failures}) *****")
            print(f"Failure Analysis: {analysis}")
            log_entry = {
                "iteration": iteration,
                "question": q_data['question'],
                "a1_internal_approach": q_data['internal_approach'],
                "a1_expected_answer": q_data['expected_answer'],
                "a2_provided_answer": a2_resp['answer'],
                "a2_provided_reasoning": a2_resp['reasoning'],
                "failure_analysis": analysis
            }
            failure_log.append(log_entry)
        elif eval_result.get('result') == 'Success':
            print("----- SUCCESS ----- (A2's independent approach was valid)")
        else:
            # Should not happen if evaluate_independent_solution validation is correct
             print(f"WARNING: Unknown evaluation result '{eval_result.get('result')}' in iteration {iteration}.")


        print(f"{'='*40}") # Separator for iterations

    # --- Loop End ---

    if iteration >= max_iterations and failure_count < target_failures:
        print(f"\nWarning: Reached max iterations ({max_iterations}) before reaching target failures ({target_failures}).")
    elif failure_count >= target_failures:
        print(f"\nTarget number of failures ({target_failures}) reached.")

    print("\nFramework run finished.")
    return failure_log

# --- Example Data & Main Execution Block ---

TEN_K_EXCERPT = """
**CONSOLIDATED STATEMENTS OF OPERATIONS**
(In millions, except per share data)
Year Ended December 31,             2023      2022      2021
----------------------------------------------------------------
Revenue                             $1200     $1000     $900
Cost of Goods Sold                   $700      $600     $550
Gross Profit                         $500      $400     $350
Research & Development               $100       $80      $70
Selling, General & Administrative    $100       $70      $60
Total Operating Expenses             $200      $150     $130
Operating Income                     $300      $250     $220
Interest Expense, net                $50       $40      $35
Other Income (Expense), net          $10       ($5)      $2
Income Before Income Taxes           $260      $205     $187
Income Tax Expense                   $52       $41      $37
Net Income                           $208      $164     $150

**CONSOLIDATED BALANCE SHEETS**
(In millions)
As of December 31,                  2023      2022      2021
----------------------------------------------------------------
Assets
Current Assets:
  Cash and cash equivalents          $150      $120     $100
  Accounts receivable, net           $350      $300     $280
  Inventory                          $200      $180     $160
  Prepaid expenses & other           $100       $80      $70
  Total current assets               $800      $680     $610
Non-current Assets:
  Property, Plant & Equipment, net   $1200     $1000     $950
  Goodwill                           $300      $300     $300
  Intangible assets, net             $100       $90      $80
  Other non-current assets           $100      $130     $110
  Total non-current assets          $1700     $1520    $1440
Total Assets                        $2500     $2200    $2050

Liabilities and Equity
Current Liabilities:
  Accounts payable                   $180      $150     $140
  Accrued expenses                   $120      $100      $90
  Short-term debt                    $50       $30      $20
  Current portion of long-term debt $100       $90      $80
  Total current liabilities          $450      $370     $330
Non-current Liabilities:
  Long-Term Debt, excluding current  $500      $410     $370
  Deferred tax liabilities           $50       $40      $35
  Other non-current liabilities      $50       $30      $25
  Total non-current liabilities      $600      $480     $430
Total Liabilities                   $1050      $850     $760
Stockholders' Equity:
  Common stock                       $10       $10      $10
  Additional paid-in capital         $400      $390     $385
  Retained earnings                  $1040     $950     $895
  Accumulated other comprehensive loss($(-10))   $(-5)     $0
  Total stockholders' equity        $1450     $1350    $1290
Total Liabilities and Equity        $2500     $2200    $2050

**Note 7: Debt**
Long-Term Debt consists primarily of senior notes due 2030. The current portion represents payments due within the next twelve months. Total Debt is the sum of Short-term debt, Current portion of long-term debt, and Long-Term Debt, excluding current portion.
For 2023: Total Debt = $50 + $100 + $500 = $650 million.
"""

if __name__ == "__main__":
    # Basic check if the key was configured earlier
    if not genai.config or not genai.config.api_key:
        print("API Key not configured. Exiting.")
    else:
        print("\n" + "="*50)
        print(" Starting Multi-Agent Framework Run with Google Gemini ")
        print("="*50)

        TARGET_FAILURES = 2 # Set how many failures you want to observe
        MAX_ITERATIONS = 5 # Set a limit to prevent excessive API calls
        failure_report = run_framework(
            TEN_K_EXCERPT,
            target_failures=TARGET_FAILURES,
            max_iterations=MAX_ITERATIONS
            )

        print("\n" + "="*50)
        print(f" Failure Report (Logged {len(failure_report)} failures) ")
        print("="*50)

        if not failure_report:
            print("No failures meeting the criteria were logged within the iteration limit.")
        else:
            for i, failure in enumerate(failure_report):
                print(f"\n--- Failure #{i+1} (Iteration: {failure['iteration']}) ---")
                print(f"  Question: {failure['question']}")
                print(f"  A1 Internal Approach: {failure['a1_internal_approach']}")
                print(f"  A1 Expected Answer: {failure['a1_expected_answer']}")
                print(f"  A2 Provided Answer: {failure['a2_provided_answer']}")
                print(f"  A2 Provided Reasoning:\n    {failure['a2_provided_reasoning'].replace(chr(10), chr(10)+'    ')}") # Indent reasoning
                print(f"  Failure Analysis: {failure['failure_analysis']}")

        print("\n" + "="*50)
        print(" Framework Simulation Complete ")
        print("="*50)