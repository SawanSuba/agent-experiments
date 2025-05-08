import os
import json
import time
import re
import csv
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Set, Dict, Any, List, Optional # Added Optional for clarity

# --- LlamaIndex specific imports ---
from llama_index.core import Document # Document is used
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.file import PDFReader

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()
print("Attempting to load environment variables...")

# Configure API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    print("Please ensure it's set in your environment or a .env file.")
    exit()
else:
    print("Google API Key found.")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not found.")
    print("This may not be an issue if you are not using any OpenAI models.")
else:
    print("OpenAI API Key found.")

# Define the models to use via LlamaIndex
# IMPORTANT: Please verify these model IDs are correct and accessible with your API key.
# The "2.5" and "2.0" versions seem unusual for Gemini as of mid-2024.
# Common Google GenAI models are like "models/gemini-1.5-pro-latest" or "models/gemini-1.5-flash-latest"
A1_MODEL_ID = "models/gemini-2.5-pro-preview-05-06" # <<< VERIFY THIS MODEL ID
A2_MODEL_ID = "models/gemini-2.0-flash" # <<< VERIFY THIS MODEL ID
print(f"Using A1 Model ID: {A1_MODEL_ID}")
print(f"Using A2 Model ID: {A2_MODEL_ID}")
print("Warning: Ensure the above model IDs are valid. If you encounter 'Model not found' errors, please update them.")


PDF_FILENAME = "aptiv.pdf" # <<< RENAME this to your PDF filename if needed

# --- Derive a clean base name for output files ---
PDF_BASE_NAME = Path(PDF_FILENAME).stem.replace(' ', '_').replace('.', '_')
print(f"Using base name '{PDF_BASE_NAME}' for log and report files.")
# --- End Derived Name Setup ---


# Generation configuration (applied during LLM instantiation)
TEMPERATUREA1 = 0.3
TEMPERATUREA2 = 0.3

# --- Setup Logging ---
LOG_FILENAME = f"{PDF_BASE_NAME}_framework_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
LOG_FIELDNAMES = [
    'Timestamp', 'Iteration', 'A1_Model', 'A2_Model', 'Question',
    'A1_Internal_Approach', 'A1_Expected_Answer', 'A2_Provided_Answer',
    'A2_Provided_Reasoning', 'Evaluation_Result', 'Failure_Analysis'
]

def initialize_log_file(filename: str, fieldnames: List[str]):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()
            print(f"Initialized log file: {filename}")

def write_log_entry(filename: str, fieldnames: List[str], log_data: Dict[str, Any]):
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        entry = {field: log_data.get(field, '') for field in fieldnames}
        writer.writerow(entry)
# --- End Logging Setup ---


# --- Helper Function for Robust JSON Extraction ---
def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a string, handling potential markdown code blocks.
    (Handles ```json ... ``` or just { ... })
    """
    # Try to find ```json ... ```
    match_markdown = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.MULTILINE)
    if match_markdown:
        json_str = match_markdown.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from markdown block. Error: {e}\nExtracted String:\n---\n{json_str}\n---")
            return None

    # If no markdown, try to find any {...}
    match_curly = re.search(r"(\{.*?\})", text, re.DOTALL | re.MULTILINE)
    if match_curly:
        json_str = match_curly.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from curly braces. Error: {e}\nExtracted String:\n---\n{json_str}\n---")
            # Fallback: try parsing the whole text if it looks like JSON
            cleaned_text = text.strip()
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                try:
                    print("Warning: No JSON markers found, attempting to parse entire response as JSON.")
                    return json.loads(cleaned_text)
                except json.JSONDecodeError as e_full:
                    print(f"Error: Also failed to parse entire response as JSON. Error: {e_full}\nResponse Text:\n---\n{text}\n---")
                    return None
            return None # Did not parse from curly braces, and not trying full text

    # If no markers, and it's not just a JSON object, try parsing the whole thing as a last resort
    cleaned_text = text.strip()
    if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
        try:
            print("Warning: No JSON markers found, attempting to parse entire response as JSON (final attempt).")
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"Error: Could not find JSON block or parse the full response.\nResponse Text:\n---\n{text}\n---")
            return None
    else:
        print(f"Error: Could not find JSON block and response doesn't appear to be raw JSON.\nResponse Text:\n---\n{text}\n---")
        return None

# --- LLM API Call Function (MODIFIED for Retries) ---
def call_llm(llm_instance: LLM, prompt: str, model_name_for_log: str,
             max_retries: int = 1, retry_delay_seconds: int = 5) -> Optional[str]:
    """
    Calls the specified LlamaIndex LLM instance with the given prompt,
    with retry logic.
    """
    print(f"\n--- Calling LlamaIndex LLM ({model_name_for_log}) ---")
    # print(f"Prompt:\n{prompt}\n--- End Prompt ---") # Uncomment for debugging

    for attempt in range(max_retries):
        try:
            response = llm_instance.complete(prompt)
            response_text = response.text
            # print(f"--- LLM Response (Raw) ---") # Often too verbose
            # print(response_text)
            # print("--- End Raw Response ---")
            return response_text
        except Exception as e:
            print(f"Error calling LlamaIndex LLM {model_name_for_log} on attempt {attempt + 1}/{max_retries}: {e}")
            if "API key" in str(e).lower():
                print("API key error detected. Retrying may not help. Please check your API key configuration.")
                # Optionally, break or return None immediately for API key errors
            if "model not found" in str(e).lower():
                print(f"Model '{model_name_for_log}' not found. Please verify the model ID. Retrying will not help.")
                return None # No point retrying if model doesn't exist
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                print(f"All {max_retries} retries failed for {model_name_for_log}.")
    return None



# --- Agent Functions ---

# *** REVISED AGGRESSIVE PROMPT FOR A1 - STRONGER EMPHASIS ON NO ASSUMPTIONS ***
def generate_question_internal_solution(llm_a1: LLM, excerpt_text: str, previous_questions: Set[str]) -> Optional[Dict[str, Any]]:
    """
    Agent A1: Generates an exceptionally challenging, multi-layered financial question
    requiring deep synthesis of information from multiple, disparate locations within the excerpt,
    WITHOUT MAKING ANY ASSUMPTIONS.
    """
    avoid_list_str = ""
    if previous_questions:
        avoid_list_str = "\n\nCRITICAL: Avoid generating questions similar to these previous ones:\n"
        avoid_list_str += "\n".join(f"- \"{q}\"" for q in previous_questions)

    prompt = f"""
You are a **Chief Financial Puzzle Master (Agent A1)**. Your mission is to craft a single, exceptionally intricate and demanding **financial question** based *only* on the provided financial document excerpt.

**YOUR ABSOLUTE, OVERRIDING, #1 CRITICAL CONSTRAINT: ZERO ASSUMPTIONS.**
*   Your question and its entire solution path MUST NOT, under any circumstances, involve any hypothetical scenarios, external knowledge, information not explicitly stated, or any form of assumption about data that isn't directly present or derivable from the provided excerpt.
*   If a piece of information is not in the text, you CANNOT use it or base any part of your question or solution on it.
*   **Examples of FORBIDDEN assumptions:**
    *   "Assuming a standard tax rate of X%..." (UNLESS the text specifies this rate for this context).
    *   "If the company had Y instead of Z..."
    *   "Assuming all 'Other Expenses' are of a certain type..." (UNLESS detailed in the text).
    *   "Extrapolating trends beyond the data provided..."
    *   "Using common industry averages or ratios not defined in the text..."
*   Every single number, rule, and step in your internal approach MUST be directly traceable to a statement, figure, or definition within the provided excerpt.

Your question must be designed to severely test an advanced AI's (Agent A2's) ability to:
1.  **Deeply comprehend** complex financial text, including footnotes and narrative sections.
2.  **Meticulously locate and extract** multiple, often non-obvious, numerical data points from **at least three distinctly separate, non-adjacent locations** within the excerpt.
3.  **Correctly apply** multi-step financial logic and calculations (involving at least **three distinct arithmetic operations**) *as defined or implied within the text itself*.
4.  Avoid being misled by plausible distractors or similar-sounding terms within the text.

The provided financial document excerpt is a concatenation of pages. Each page's content is clearly demarcated by '--- START OF Page X ---' and '--- END OF Page X ---' markers. You MUST use these page markers (e.g., "Page X") when referring to specific page numbers in your internal approach.

Excerpt:
\"\"\"{excerpt_text}\"\"\"
{avoid_list_str}

**FURTHER CRITICAL CONSTRAINTS (NON-NEGOTIABLE, alongside the NO ASSUMPTIONS rule):**
*   **STRICTLY EXCERPT-BASED:** All data and rules for calculation must originate *solely* from the provided text.
*   **NUMERICAL ANSWER:** The question must have a precise numerical answer (number or percentage).
*   **MULTI-LOCATION DATA IMPERATIVE:** The question *must absolutely require* data from at least **three distinctly separate sections or pages** of the excerpt. A question solvable from a single table or page is NOT acceptable.
*   **MULTI-STEP CALCULATION IMPERATIVE:** The solution must involve at least **three sequential or nested arithmetic operations**.
*   **FINANCIAL NATURE:** The question must be deeply financial, testing understanding of financial statements, ratios, metrics, or accounting principles.
*   **NO EASY LOOKUPS:** The question should NOT be answerable by simply finding one or two obvious numbers and performing a trivial calculation. It must demand significant synthesis.
*   **NO EXPLICIT LOCATION HINTS IN QUESTION:** The question itself must not mention page numbers, specific table titles, or section names that would give away data locations to Agent A2.

**Internal Pre-computation & Self-Critique (Before formulating the final JSON):**
**STEP 0: ASSUMPTION CHECK - THE GATEKEEPER.**
*   Review every part of your potential question and internal solution. Is there *any* step, data point, or logical jump that relies on information NOT explicitly present in the excerpt?
*   If YES, **IMMEDIATELY DISCARD AND RESTART your question generation process from scratch.** Do not try to patch it. This rule is paramount.

Once you are 100% certain there are NO assumptions:
1.  **Identify Candidate Data:** Scan the excerpt for at least 3-4 non-obvious numerical data points located far apart.
2.  **Devise Complex Connection:** Think of a legitimate, multi-step financial calculation or derivation that *necessitates* combining these specific, disparate data points, based *only* on rules/definitions in the text.
3.  **Consider Distractors:** Are there similar figures or terms in the text that could mislead? Your question should implicitly require A2 to navigate these carefully.
4.  **Difficulty Check (while ensuring NO assumptions):**
    *   Is this question genuinely hard *without resorting to assumptions*?
    *   Does it force A2 to read footnotes or narrative text in conjunction with tables?
    *   Is the required data "hidden" in plain sight or embedded in dense text?
    *   Could a human analyst solve it without very careful, meticulous work across multiple pages, using only the text?
    *   If the question feels too simple (even if assumption-free), try to increase complexity by requiring more *text-based* data points or more *text-based* calculation steps.

**Task (Only if the question passes the rigorous NO ASSUMPTION check):**
1.  Generate *one* ***NEW***, exceptionally challenging, multi-step numerical **financial question** adhering to ALL constraints above.
2.  Internally determine the precise, detailed, step-by-step approach.
    *   Start with "Data Source Origin: Multiple distinct locations/pages (specify number, e.g., 3 distinct locations on Pages X, Y, Z)".
    *   For each step, cite the **exact numerical value, its source page label (e.g., 'Page 5'), and context (e.g., 'from table "Consolidated Income Statements", line "Net Revenues" for 2023')**.
    *   Explain the financial logic for each calculation step, ensuring this logic is derived from the text.
3.  Calculate the precise expected numerical answer, using only text-derived data.

**Output Format:**
Respond *only* with a single, valid JSON object: {{"question": "...", "internal_approach": "...", "expected_answer": "..."}}

**Example of a "Harder" Question (Illustrative - create your own unique one, 100% based on text):**
{{
  "question": "Calculate the company's 'Cash Flow from Operations to Capital Expenditures' ratio for 2023, where Capital Expenditures are defined in Note 4 as 'Purchases of property, plant, and equipment' plus 'Acquisitions of intangible assets'. Both 'Net cash provided by operating activities' and the components of Capital Expenditures must be sourced from the provided financial statements and their accompanying notes.",
  "internal_approach": "Data Source Origin: Multiple distinct locations/pages (e.g., Cash Flow Statement on Page 50, Note 4 on Page 68).\\nFinancial Concept: Ratio = Net Cash from Operations / (Purchases of PP&E + Acquisitions of Intangibles), with definition of CapEx from text.\\n1. Find 'Net cash provided by operating activities' for 2023 ($300M) from 'Consolidated Statements of Cash Flows', Page 50.\\n2. Find 'Purchases of property, plant, and equipment' for 2023 ($80M) from 'Note 4: Property, Plant, and Equipment', section detailing capital expenditures, Page 68.\\n3. Find 'Acquisitions of intangible assets' for 2023 ($20M) from 'Note 4: Property, Plant, and Equipment' (or a relevant Intangibles note if separate), section detailing capital expenditures, Page 68. (If this value is zero or not separately listed but the definition includes it, this must be noted from the text, e.g., 'Acquisitions of intangible assets for 2023 ($0M) as per Note X which shows no such activity for the period').\\n4. Calculate Total Capital Expenditures for 2023: $80M + $20M = $100M.\\n5. Calculate the Ratio: $300M / $100M.",
  "expected_answer": "3.0"
}}
"""
    response_text = call_llm(llm_a1, prompt, A1_MODEL_ID, max_retries=1) # Still keeping retries low for A1 generation
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'question' in data and 'internal_approach' in data and 'expected_answer' in data:
        print("A1 Generation Successful (JSON Parsed).")
        if "assumption" in data.get('question','').lower() or "assume" in data.get('question','').lower():
            print(f"CRITICAL WARNING: A1's question may still contain 'assume' or 'assumption': {data.get('question')}")
        if "assumption" in data.get('internal_approach','').lower() or "assume" in data.get('internal_approach','').lower():
            print(f"CRITICAL WARNING: A1's internal_approach may still contain 'assume' or 'assumption': {data.get('internal_approach')[:300]}...")
        
        # Basic checks for the difficulty requirements (can be expanded)
        if "Data Source Origin: Multiple distinct locations/pages" not in data.get('internal_approach', ''):
            print("Warning: A1's internal_approach might not explicitly state 'Multiple distinct locations/pages' as required.")
        page_refs = re.findall(r"Page \w+", data.get('internal_approach', ''), re.IGNORECASE)
        if len(set(page_refs)) < 2 and "Multiple distinct locations/pages" in data.get('internal_approach', ''):
             print(f"Warning: A1 claims multiple pages but internal_approach only has {len(set(page_refs))} unique page references. Approach: {data.get('internal_approach', '')[:200]}...")
        return data
    else:
        print(f"Error: A1 generation response parsing failed or missing required keys.")
        if data:
            print(f"Missing keys from {{'question', 'internal_approach', 'expected_answer'}}. Found: {list(data.keys())}")
        else:
            print(f"Raw response from A1 that failed parsing or extraction: {response_text[:500]}...")
        return None

# (attempt_solution_independent function - MODIFIED to use retries for A2)
def attempt_solution_independent(llm_a2: LLM, excerpt_text: str, question: str) -> Optional[Dict[str, Any]]:
    """
    Agent A2: Attempts to answer the question independently using its LLM.
    Uses retry logic for the LLM call.
    """
    prompt = f"""
You are an analyst agent (Agent A2). You need to answer a question based *only* on the provided financial document excerpt. You must figure out the method yourself using only the information given.
The provided financial document excerpt may be a concatenation of pages. Each page's content might be demarcated by '--- START OF Page X ---' and '--- END OF Page X ---' markers. You can use this information if helpful but your primary focus is the content.

Given the following financial document excerpt:
\"\"\"{excerpt_text}\"\"\"

Answer the following question *strictly based on the excerpt provided*:
"{question}"

Instructions:
1.  Carefully analyze the question and the excerpt.
2.  Devise your own step-by-step method to find the relevant information *only within the excerpt* and perform any necessary calculations. Clearly state the data points you are using from the text. If you identify data from specific pages based on markers like 'Page X', you can mention it.
3.  Provide your final numerical answer. Ensure the final answer format is a simple number or percentage where appropriate (e.g., "0.5", "50%", "210.2").
4.  Clearly explain the reasoning and the exact steps you took to arrive at your answer, referencing data found *only* in the excerpt.

Output Format:
Respond *only* with a single, valid JSON object containing the keys 'answer' and 'reasoning'. Do not include any introductory text, explanations, comments, or markdown formatting outside the JSON object itself.

Example JSON Output:
{{
  "answer": "0.72",
  "reasoning": "1. Identified Total Liabilities for 2023 as $1050 million from the Balance Sheet section in the provided text (found on Page 15 of excerpt). 2. Identified Total stockholders' equity for 2023 as $1450 million from the Balance Sheet section (found on Page 15 of excerpt). 3. Calculated Debt-to-Equity ratio as Total Liabilities / Total Equity = $1050 / $1450 = 0.7241. Rounded to 0.72."
}}
"""
    # A2's task is more prone to transient issues or needing a second thought.
    response_text = call_llm(llm_a2, prompt, A2_MODEL_ID, max_retries=3, retry_delay_seconds=7)
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and 'answer' in data and 'reasoning' in data:
        print("A2 Solution Attempt Successful (JSON Parsed).")
        return data
    else:
        print(f"Error: A2 solving response parsing failed or missing required keys.")
        if data:
            print(f"Missing keys from {{'answer', 'reasoning'}}. Found: {list(data.keys())}")
        else:
            print(f"Raw response from A2 that failed parsing or extraction: {response_text[:500]}...")
        return None

# (evaluate_independent_solution function remains unchanged)
def evaluate_independent_solution(llm_a1: LLM, question_data: dict, a2_response: dict) -> Optional[Dict[str, Any]]:
    """
    Agent A1: Evaluates A2's independent solution attempt using its LLM.
    """
    def _to_float(value_str):
        try:
            cleaned_str = str(value_str).replace('$', '').replace(',', '').replace('%', '').strip()
            return float(cleaned_str)
        except (ValueError, TypeError):
            return None

    a1_answer_float = _to_float(question_data.get('expected_answer'))
    a2_answer_float = _to_float(a2_response.get('answer'))
    numerical_match_flag = "Could not compare numerically"
    if a1_answer_float is not None and a2_answer_float is not None:
        # Allow for small tolerance
        is_close = abs(a1_answer_float - a2_answer_float) < 0.02 * abs(a1_answer_float) if a1_answer_float != 0 else abs(a2_answer_float) < 0.02
        is_close = is_close or abs(a1_answer_float - a2_answer_float) < 0.2 # Absolute tolerance for small numbers
        numerical_match_flag = "Yes" if is_close else "No"
    elif str(question_data.get('expected_answer', '')).strip().lower() == str(a2_response.get('answer', '')).strip().lower():
        numerical_match_flag = "Yes (Exact String Match)"


    prompt = f"""
You are the expert financial analyst (Agent A1) who originally created a question and determined the correct approach/answer based *only* on a specific financial document excerpt. You are now evaluating an attempt by another agent (Agent A2) who had to devise their *own* method to solve the question using *only* the same excerpt.

Your Original Question:
"{question_data.get('question', 'N/A')}"

Your Internal Step-by-Step Approach (this shows where YOU found the data, including page numbers like 'Page X' from the processed excerpt, and if it was from single/multiple pages):
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
    * Did A2 devise a *logically valid* method to arrive at *an* answer using data plausibly found in the original excerpt? (Focus on logical flow, not whether it matches your exact method or your page references).
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
  "analysis": "A2 correctly identified the need for Total Liabilities and Equity but used the 2022 figures ($850/$1350) instead of the required 2023 figures ($1050/$1450) from the excerpt, leading to an incorrect numerical answer. A2 mentioned finding data on 'Page 10' while my internal solution used 'Page 15', but this difference in A2's page reference is acceptable if A2's chosen data was otherwise correct for the question's parameters."
}}
"""
    # Evaluation is also critical, fewer retries.
    response_text = call_llm(llm_a1, prompt, f"{A1_MODEL_ID} (Evaluation)", max_retries=1)
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
        if data:
            print(f"Missing key 'result'. Found: {list(data.keys())}")
        else:
            print(f"Raw response from A1 (Evaluation) that failed parsing or extraction: {response_text[:500]}...")
        return None


# --- Orchestrator Function (MODIFIED PDF Loading) ---
def run_framework(pdf_path: Path, llm_a1: LLM, llm_a2: LLM, target_failures: int = 2, max_iterations: int = 5) -> List[Dict[str, Any]]:
    """
    Orchestrates the interaction between A1 and A2 using LlamaIndex LLMs,
    logging all iterations to CSV and collecting failures, avoiding repeated questions.
    PDF text is pre-processed to include page markers.
    """
    failure_count = 0
    failure_log_list = []
    iteration = 0
    previous_questions: Set[str] = set()

    # --- Load PDF Content (MODIFIED to include page markers) ---
    pdf_file_str = str(pdf_path)
    print(f"\nLoading financial excerpt from: {pdf_file_str}")
    if not pdf_path.is_file():
        print(f"Error: PDF file not found at {pdf_file_str}")
        return []
    try:
        reader = PDFReader()
        documents: List[Document] = reader.load_data(file=pdf_path) # type hint for clarity
        if not documents:
            print(f"Error: No content could be extracted from {pdf_file_str}")
            return []

        page_texts = []
        for doc in documents:
            # page_label should be like '1', '2', 'iii', etc.
            page_label = doc.metadata.get('page_label', f"UnknownPage_{len(page_texts)+1}")
            file_name = doc.metadata.get('file_name', pdf_path.name) # Get file_name from metadata
            
            # Using a simpler marker for LLM processing
            header = f"--- START OF PAGE {page_label} FROM {file_name} ---"
            footer = f"--- END OF PAGE {page_label} FROM {file_name} ---"
            page_texts.append(f"{header}\n{doc.text}\n{footer}")

        excerpt_text = "\n\n".join(page_texts)
        print(f"Successfully loaded and extracted text from {len(documents)} page(s) of PDF (total length: {len(excerpt_text)} chars).")
        if len(excerpt_text) < 100: # Check combined length
            print("Warning: Extracted text seems very short. Ensure the PDF contains sufficient data.")
        elif len(documents) == 1 and len(excerpt_text) > 200000: # Heuristic for single large document
             print(f"Warning: PDF loaded as a single document of {len(excerpt_text)} characters. Page-specific features might be limited if not internally structured.")

    except Exception as e:
        print(f"Error reading or parsing PDF file {pdf_file_str}: {e}")
        return []
    # --- PDF Loading End ---

    print(f"\nStarting framework run. Target Failures: {target_failures}, Max Iterations: {max_iterations}")
    print(f"Logging iterations to: {LOG_FILENAME}")
    initialize_log_file(LOG_FILENAME, LOG_FIELDNAMES)

    while failure_count < target_failures and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*15} Iteration {iteration} {'='*15}")
        time.sleep(1) # Reduced sleep, LLM calls have inherent delays

        q_data = None
        a2_resp = None
        eval_result = None
        current_log_data: Dict[str, Any] = { # type hint for clarity
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Iteration': iteration,
            'A1_Model': A1_MODEL_ID,
            'A2_Model': A2_MODEL_ID,
            'Question': 'GENERATION FAILED',
            'A1_Internal_Approach': '', 'A1_Expected_Answer': '',
            'A2_Provided_Answer': 'GENERATION FAILED', 'A2_Provided_Reasoning': '',
            'Evaluation_Result': 'N/A', 'Failure_Analysis': 'A1 Generation Failed'
        }

        print("\nStep 1: A1 generating question and internal solution...")
        q_data = generate_question_internal_solution(llm_a1, excerpt_text, previous_questions)

        if q_data:
            generated_question = q_data.get('question', 'N/A')
            print(f"A1 Generated Question: {generated_question}")
            previous_questions.add(generated_question)
            current_log_data.update({
                'Question': generated_question,
                'A1_Internal_Approach': q_data.get('internal_approach', 'N/A'),
                'A1_Expected_Answer': q_data.get('expected_answer', 'N/A'),
                'A2_Provided_Answer': 'ATTEMPT SKIPPED', 'A2_Provided_Reasoning': '',
                'Evaluation_Result': 'N/A', 'Failure_Analysis': ''
            })
        else:
            print(f"ERROR: Iteration {iteration} - A1 generation failed.")
            write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
            continue

        print("\nStep 2: A2 attempting independent solution...")
        a2_resp = attempt_solution_independent(llm_a2, excerpt_text, q_data['question'])
        if a2_resp:
            print(f"A2 Answer: {a2_resp.get('answer', 'N/A')}")
            # print(f"A2 Reasoning:\n{a2_resp.get('reasoning', 'N/A')}") # Can be verbose
            current_log_data.update({
                'A2_Provided_Answer': a2_resp.get('answer', 'N/A'),
                'A2_Provided_Reasoning': a2_resp.get('reasoning', 'N/A'),
                'Evaluation_Result': 'EVALUATION PENDING', 'Failure_Analysis': ''
            })
        else:
            print(f"ERROR: Iteration {iteration} - A2 solving failed.")
            current_log_data.update({
                'A2_Provided_Answer': 'SOLVING FAILED', 'A2_Provided_Reasoning': '',
                'Evaluation_Result': 'N/A', 'Failure_Analysis': 'A2 Solving Failed'
            })
            write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
            continue # Skip evaluation if A2 failed to produce a response

        print("\nStep 3: A1 evaluating A2's independent solution...")
        eval_result = evaluate_independent_solution(llm_a1, q_data, a2_resp)
        if eval_result:
            print(f"Evaluation Result: {eval_result.get('result', 'N/A')}")
            if eval_result.get('result') == 'Failure':
                print(f"Failure Analysis by A1: {eval_result.get('analysis', 'N/A')}")
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
            # Log entry will be written below regardless of eval success/failure if q_data and a2_resp existed

        write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)

        if eval_result and eval_result.get('result') == 'Failure':
            failure_count += 1
            analysis = eval_result.get('analysis', 'No analysis provided by evaluator.')
            print(f"***** FAILURE LOGGED ({failure_count}/{target_failures}) *****")
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
        # elif not eval_result and (q_data and a2_resp): # This case is handled by the eval_result check above
        #     print("----- ITERATION ENDED (Evaluation Error) -----")
        # No need for these extra prints, covered by ERROR messages and final log
        print(f"{'='*40}")

    if iteration >= max_iterations and failure_count < target_failures:
        print(f"\nWarning: Reached max iterations ({max_iterations}) before reaching target failures ({target_failures}).")
    elif failure_count >= target_failures:
        print(f"\nTarget number of failures ({target_failures}) reached.")

    print(f"\nFramework run finished. Detailed log available in: {LOG_FILENAME}")
    return failure_log_list


# --- Main Execution Block ---
if __name__ == "__main__":
    if not google_api_key:
        # Already handled by print and exit at the top, but good for explicitness
        print("Google API Key not configured properly. Exiting.")
        exit()

    print("\n" + "="*50)
    print(" Starting Multi-Agent Framework Run with LlamaIndex ")
    print("="*50)

    try:
        print(f"Instantiating A1 LLM: {A1_MODEL_ID} with temp {TEMPERATUREA1}")
        llm_a1 = GoogleGenAI(
            model_name=A1_MODEL_ID,
            api_key=google_api_key,
            temperature=TEMPERATUREA1
        )
        print(f"Instantiating A2 LLM: {A2_MODEL_ID} with temp {TEMPERATUREA2}")
        llm_a2 = GoogleGenAI(
            model_name=A2_MODEL_ID,
            api_key=google_api_key,
            temperature=TEMPERATUREA2
        )
        print("LLM instances created successfully.")
    except Exception as e:
        print(f"FATAL: Failed to instantiate LLMs: {e}")
        if "model not found" in str(e).lower() or "could not find model" in str(e).lower():
            print(f"Please verify that the model IDs ('{A1_MODEL_ID}', '{A2_MODEL_ID}') are correct and accessible with your API key.")
            print("Common Google GenAI model names are like 'models/gemini-1.5-pro-latest' or 'models/gemini-1.5-flash-latest'.")
        exit()

    TARGET_FAILURES = 2
    MAX_ITERATIONS = 5 # Set to a small number for testing, increase as needed

    pdf_file_path = Path(PDF_FILENAME)

    if not pdf_file_path.is_file():
        print(f"\nFATAL ERROR: PDF file '{PDF_FILENAME}' not found in the current directory ({Path.cwd()}).")
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
            report_filename = f"{PDF_BASE_NAME}_failure_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
                # print(f"  A1 Approach: {failure['a1_internal_approach']}") # Can be long
                # print(f"  A2 Reasoning: {failure['a2_provided_reasoning']}") # Can be long
                print(f"  A1 Expected: {failure['a1_expected_answer']}, A2 Provided: {failure['a2_provided_answer']}")
                print(f"  Failure Analysis: {failure['failure_analysis']}")

    print("\n" + "="*50)
    print(" Framework Simulation Complete ")
    print("="*50)