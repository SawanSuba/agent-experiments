import os
import json
import time
import re
import csv
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Set, Dict, Any, List, Optional

# --- LlamaIndex specific imports ---
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.file import PDFReader

# --- Configuration ---
load_dotenv()
print("Attempting to load environment variables...")

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    exit()
else:
    print("Google API Key found.")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not found.")
else:
    print("OpenAI API Key found.")

A1_MODEL_ID = os.getenv("A1_MODEL_ID", "models/gemini-2.5-pro-preview-05-06") # Changed default
A2_MODEL_ID = os.getenv("A2_MODEL_ID", "models/gemini-2.0-flash") # Changed default
print(f"Using A1 Model ID: {A1_MODEL_ID}")
print(f"Using A2 Model ID: {A2_MODEL_ID}")

PDF_FILENAME = "copart.pdf" # <<< RENAME this to your PDF filename if needed



PDF_BASE_NAME = Path(PDF_FILENAME).stem.replace(' ', '_').replace('.', '_')
print(f"Using base name '{PDF_BASE_NAME}' for log and report files.")

TEMPERATUREA1 = 0.5 # Adjusted slightly as per original A1 complex prompt suggestion
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

# --- Helper Function for Robust JSON Extraction ---
def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    match_markdown = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.MULTILINE)
    if match_markdown:
        json_str = match_markdown.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from markdown block. Error: {e}\nExtracted String:\n---\n{json_str}\n---")
            return None

    match_curly = re.search(r"(\{.*?\})", text, re.DOTALL | re.MULTILINE)
    if match_curly:
        json_str = match_curly.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from curly braces. Error: {e}\nExtracted String:\n---\n{json_str}\n---")
            cleaned_text = text.strip()
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                try:
                    print("Warning: No JSON markers found, attempting to parse entire response as JSON.")
                    return json.loads(cleaned_text)
                except json.JSONDecodeError as e_full:
                    print(f"Error: Also failed to parse entire response as JSON. Error: {e_full}\nResponse Text:\n---\n{text}\n---")
                    return None
            return None
    
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

# --- LLM API Call Function ---
def call_llm(llm_instance: LLM, prompt: str, model_name_for_log: str,
             max_retries: int = 1, retry_delay_seconds: int = 5) -> Optional[str]:
    print(f"\n--- Calling LlamaIndex LLM ({model_name_for_log}) ---")
    for attempt in range(max_retries):
        try:
            response = llm_instance.complete(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling LlamaIndex LLM {model_name_for_log} on attempt {attempt + 1}/{max_retries}: {e}")
            if "API key" in str(e).lower():
                print("API key error. Retrying may not help.")
            if "model not found" in str(e).lower():
                print(f"Model '{model_name_for_log}' not found. Retrying will not help.")
                return None
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                print(f"All {max_retries} retries failed for {model_name_for_log}.")
    return None

# --- Agent Functions ---
def generate_question_internal_solution(llm_a1: LLM, excerpt_text: str, previous_questions: Set[str]) -> Optional[Dict[str, Any]]:
    """
    Agent A1: Generates an exceptionally challenging financial question, a detailed internal solution
    (as an array of strings and a metadata dictionary), and the expected answer,
    based ONLY on the provided excerpt and WITHOUT MAKING ANY ASSUMPTIONS.
    The solution steps and metadata MUST adhere to the specified QA formatting guidelines.
    """
    avoid_list_str = ""
    if previous_questions:
        avoid_list_str = "\n\nCRITICAL: Avoid generating questions similar to these previous ones:\n"
        avoid_list_str += "\n".join(f"- \"{q}\"" for q in previous_questions)

    # This is the detailed prompt incorporating all QA feedback for A1's generation task
    prompt = f"""
You are an expert financial analyst (Agent A1). Your primary goal is to generate exceptionally challenging **financial questions** based *only* on the provided financial document excerpt. These questions are designed to rigorously test another language model's ability to perform complex calculations, meticulously extract data from disparate parts of the text, and apply nuanced financial rules strictly from the document.

The provided financial document excerpt is a concatenation of pages. Each page's content is clearly demarcated by '--- START OF Page X ---' and '--- END OF Page X ---' markers, where 'X' is the page number or label. You MUST use these page markers (e.g., "Page X") when referring to specific page numbers in your internal approach.

Based *only* on the provided financial document excerpt:
\"\"\"{excerpt_text}\"\"\"
{avoid_list_str}

*** IMMEDIATE AND CRITICAL CONSTRAINT: STOP and review your generated question before outputting. It ABSOLUTELY MUST NOT involve any hypothetical scenarios, assumptions, or information not explicitly stated in the provided excerpt. ***
Avoid any question that starts with or relies on phrases like:
- "Assuming..."
- "If the company were to..."
- "What if..."
- "If X changes to Y..."
- "Assuming a different interest rate of..."
- "If we exclude..." (unless 'excluding' a specifically identifiable item mentioned *within* the text's structure, like excluding a specific line item from a list or total *given in the text*)
- Questions requiring outside knowledge or external data not present in the text.

The question must be a direct inquiry about a value or relationship that is *provably* derivable by combining or calculating with data and rules *explicitly present* in the given text excerpt, and *only* the given text excerpt. The challenge must come from the complexity of finding, extracting, and calculating with data *within* the excerpt, not from making assumptions or using outside information.

+ **CRITICAL PRELIMINARY INTERNAL ANALYSIS (Before Generating the Question):**
+ Your *first internal step*, before attempting to formulate the question itself, MUST be to thoroughly analyze the entire provided `excerpt_text`.
+ The goal of this preliminary analysis is to identify one or more **"prospect data sets"** – these are combinations of information within the excerpt that have high potential for creating a genuinely complex, multi-step, model-breaking question according to the criteria outlined further below.
+ **What to look for in a "prospect data set" during your internal analysis:**
+   1.  **Multiple, Interrelated Data Points:** Can you find distinct pieces of information (numerical values, categories, dates, conditions) that are spread across different parts of the excerpt (e.g., different tables, narrative sections on different pages, footnotes) but are logically connectable?
+   2.  **Potential for Multi-Attribute Filtering:** Does the data allow for filtering items based on *several simultaneous criteria*? For example:
+       *   Identifying items belonging to specific, multiple categories (e.g., "senior *and* junior unsecured notes").
+       *   Filtering by a specific status (e.g., "issued *and* outstanding").
+       *   Applying date-based conditions on different attributes (e.g., "issued after 2020" *and* "maturing before 2050").
+   3.  **Sufficient Data for Calculation:** If items are filtered using such complex criteria, are there associated numerical values (e.g., principal amounts, interest rates, quantities, monetary values) that can then be used in a multi-step calculation (e.g., summing values, calculating a weighted average, finding a net change, determining total payout)?
+   4.  **Implicit Rules/Definitions from Prose:** Are there narrative sections (e.g., definitions in footnotes, accounting policies described in text) that define how certain items should be treated or how a calculation should be performed, which can be applied to numerical data found elsewhere?
+
+ **Action Following Internal Analysis:** After this internal analysis, you should have identified the most promising areas within the excerpt to construct your challenging question. *Then, and only then*, proceed to the main task of generating the question and solution, focusing on the richest "prospect data set" you identified. This pre-analysis is crucial for maximizing the question's complexity and ensuring it is well-grounded in the provided text.

Task:
1.  Generate *one* ***NEW***, highly challenging, multi-step numerical **financial question**, drawing upon the insights from your preliminary internal analysis of "prospect data sets." This question must be of the type a financial analyst would ask when scrutinizing a company's financial statements and related disclosures. It must be specifically designed to "break" a less capable model, **while strictly adhering to the CRITICAL CONSTRAINT above (No Assumptions/Hypotheticals)**.
    *   The question *must strive to* require combining numerical data points located in *different, distinct locations within the excerpt* (e.g., different pages, different named tables that appear on different pages, a table vs. a footnote on another page) – ideally reflecting the structure of the "prospect data set" you identified.
    *   The question *must* involve a multi-step calculation (more than one arithmetic operation, e.g., add A, subtract B, then divide by C), typically performed on the data identified and filtered.
    *   The question *must* test the ability to apply a specific **financial concept, ratio, metric, or accounting rule** that is *derivable* or *described* within the text, even if not presented as a ready-made formula. The financial nature of the question is paramount.
    *   The question *must not* contain explicit references to page numbers, table names, or section titles that would give away the location of the data to the answering agent. The answering agent must find the data based on financial terminology and context.

    **Key Characteristics of Model-Breaking Questions (Inspired by Design Criteria):**
        To make the question particularly challenging and "model-breaking," ensure it incorporates elements like those you would have looked for in your preliminary analysis:
        *   **Multi-Attribute Filtering:** Require identifying and isolating financial items based on several specific criteria simultaneously, as found in your prospect data set.
        *   **Conditional Logic from Prose:** The conditions for filtering or calculation might be described in narrative text.
        *   **Targeted Calculation on Filtered Data:** After identifying the relevant items through complex filtering, the question should require a specific calculation on these items.

    *   Consider questions that involve core **financial analysis tasks** such as:
        *   Calculating specific **financial ratios** (e.g., liquidity, profitability, solvency, efficiency ratios) where components are found in different statements or footnotes.
        *   Determining **period-over-period growth rates or percentage changes** for complex financial line items or derived metrics.
        *   Reconciling or deriving a specific financial figure by **aggregating or disaggregating components** listed across multiple tables, sections, or footnotes – especially when combined with multi-attribute filtering.
        *   Applying a specific **accounting policy or financial definition** mentioned in a narrative part of the text to numerical data presented elsewhere.
        *   Calculating key performance indicators (KPIs) or non-GAAP measures if their components and calculation methodology are described *within the excerpt*.
    *   The question must be substantially different from any previous questions listed above. When generating a *NEW* question, strive for novelty not just in the specific numbers or entities involved, but also in the *combination of financial concepts tested*, the *structure of the multi-step reasoning*, or the *types of data extraction and filtering required*.
    *   The question must be *directly and unambiguously* answerable by calculation or synthesis using *only* the numerical data and statements present in the excerpt.

2.  **Provide the detailed components for an internal solution approach** for answering the generated question. This includes a `solution_array` and a `metadata_dict`.


**Your Task:**
Based **ONLY** on the provided excerpt and **WITHOUT MAKING ANY ASSUMPTIONS**, generate:
a) An exceptionally challenging financial question.
b) A `solution_array` containing the detailed step-by-step solution for this question.
c) A `metadata_dict` related to this solution.
d) The final `expected_answer`.

**Detailed Structure and Formatting Guidelines for `solution_array` and `metadata_dict`:**

**`solution_array` (List of Strings):**
Each string in this list represents a complete step, meticulously following all formatting rules.
Example structure for strings within the array:
[
    "## Step 1 Title (e.g., Process of obtaining the <required-answer>).\\n<This first substep should be an overview of the process. All substeps must start with \\n and a capital letter, and end with a full stop if it's a complete sentence. Ensure elaborative statements for clarity.>",
    "## Step 2 Title (e.g., Locate <specific table/chart name> containing <data needed>).\\n<Substep 1: Describe the search process. After searching through the report, we can find the <table/chart name> on page <PDF viewer page number from 'Page X' marker>. This table/chart contains <description of relevant data it holds>. Use PDF viewer page numbers, not internal document page numbers.>\n<Substep 2: Elaborate on the table/chart if necessary. For instance: The table titled “<Exact Table/Chart Title as in Document> (In thousands, millions, etc.)” is located on page <PDF viewer page number from 'Page X' marker>.>",
    "## Step 3 Title (e.g., Extract <required-data-1> from <table/chart name>).\\n<Substep 1: Describe the table structure. For example: The table is structured with <financial accounts/categories> listed vertically in the leftmost column, and reported figures for <specific years/periods like 2023, 2022, Q1> displayed horizontally across the subsequent columns, labeled <header for column 1>, <header for column 2>, etc.>\n<Substep 2: Pinpoint the data. To find the <required-data-1>, look at the row labeled “<Exact Row Label>”. Then, move horizontally to the column corresponding to “<Exact Column Header>”.>\n<Substep 3: State the extracted value. The <required-data-1> value listed in that cell is <value with unit>. Use exact figures as shown in the financial report.>",
    "## Step 4 Title (e.g., Extract <required-data-2> from <another table/chart or same one>).\\n<Follow similar detailed substeps as in Step 3 to locate and extract additional necessary data. Clearly state what specific information is being sought from each source and how it connects to the overall solution.>",
    "## Step 5 Title (e.g., Calculate <name of required-metric-1>).\\n<Substep 1: Define the formula in words or symbols. For example: The <name of metric-1> is calculated as (Net Income / Average Shareholder Equity) * 100.>\n<Substep 2: Input the extracted data into the formula. For example: ($1,000 / (($5,000 + $4,000) / 2)) * 100. There should be NO spaces around mathematical operators like +, -, *, /, except for spaces around the = sign.>\n<Substep 3: Show the calculation result. For example: ($1,000 / $4,500) * 100 = 22.22%. Conclude with a statement like: Therefore, the <name of metric-1> is 22.22%. Do NOT state the final overall answer here; reserve 'Thus' or 'Therefore' for the final concluding statement in the last step when referring to the overall question's answer.>",
    // (Add more calculation steps as needed, minimum 4 ## steps in total for the solution, including extraction and calculation steps. Each step must have a clear title and elaborative substeps.)
    "## Final Conclusion (e.g., Calculate the Final Answer: <specific metric being calculated>).\\n<Substep 1: Explain how the final metric is derived, potentially restating a final formula. For example: The <final-answer-metric> can be calculated using the formula: <Metric-1 result> - <Metric-2 result>.>\n<Substep 2: Show the final calculation. For example: 22.22% - 5.10% = 17.12%.>\n<Substep 3: Provide a summary statement. Thus, <summary statement restating the finding or result in response to the question posed>.>\n<Substep 4: State the single final answer. The final answer is\\n<single final numeric answer with appropriate unit, no full-stop/period after the unit>>"
]

**`metadata_dict` (Dictionary):**
{{
    "page_numbers": [<PDF viewer page no.1 from 'Page X' marker>, <PDF viewer page no.2 from 'Page X' marker>],
    "grounding_elements": ["<exact table/chart/infographic name 1>", "<exact table/chart/infographic name 2>"],
    "type": ["multi-span" OR "single-span"],
    "difficulty": ["easy" OR "medium" OR "hard"],
    "models_failed": ["o3" OR "NA"]
}}

**CRITICAL FORMATTING AND CONTENT RULES (Review all QA feedback points):**
*   **Strict Adherence:** Follow the structure, `\\n` usage, capitalization, punctuation, and spacing rules for each string in `solution_array` and for `metadata_dict` precisely as exemplified and detailed in all QA feedback.
*   **`\\n` Newlines:**
    *   Every substep within a `## Step Title` string MUST begin with `\\n`.
    *   NO spaces before or after `\\n`.
    *   The final answer in the "Final Conclusion" step's string MUST be preceded by `\\n` on its own line (e.g., `...is\\nFinal Answer Value`).
    *   The concluding summary statement in the "Final Conclusion" step's string MUST also be preceded by `\\n`.
    *   Avoid any unnecessary `\\n`.
*   **`##` Step Headers:**
    *   Each main step string in the `solution_array` MUST start with `##` followed by a space, then a descriptive title (e.g., `## Locate the Consolidated Balance Sheet.`).
    *   There should be NO leading spaces before `##`.
    *   Ensure a minimum of 4 distinct `##` step strings in the `solution_array`.
*   **Substeps:**
    *   Substeps (lines following `\\n` within a step string) MUST start with a capital letter.
    *   Substeps MUST be elaborative, providing clear context and detailed explanations.
    *   End complete sentences in substeps with a full stop.
*   **Mathematical Operations:**
    *   NO spaces around mathematical operators (`+`, `-`, `*`, `/`). For example, write `Value1*Value2/Value3`.
    *   Spaces ARE allowed before and after the equals sign (`=`).
*   **Formulas:**
    *   Within a calculation step string, first clearly define or state the formula in words or symbols as a substep. Then, show the substitution of values into the formula in the next substep.
    *   Do NOT present formulas as separate `## Formula Title` steps.
*   **Data Extraction:**
    *   When describing table/data location, be very specific: PDF viewer page number (from 'Page X' markers), exact table/chart title (if available, enclosed in double quotes), exact row labels, and exact column headers.
    *   Clearly explain how to navigate the table to find the specific data point.
    *   Use the exact numerical figures as they appear in the document.
*   **Grammar and Punctuation:** Ensure impeccable grammar and correct punctuation throughout all generated strings. No trailing commas at the end of the textual content of step titles or substeps unless grammatically part of the sentence.
*   **No Bolding/Highlighting:** Do not use any bolding or other text highlighting.
*   **Quotation Marks:** Use quotation marks (" ") only for exact titles, labels, or direct quotes from the document. Avoid excessive or unnecessary use.
*   **Page Numbers in Metadata:** Use the page numbers from the '--- START OF Page X ---' markers in the excerpt. These should be integers or strings as they appear in the markers.
*   **Final Answer Consistency:** The `expected_answer` field in the output JSON must exactly match the final answer stated in the final substep of your "## Final Conclusion" string in the `solution_array`.

**Final JSON Output Format for Your Entire Response:**
Respond *only* with a single, valid JSON object structured as follows. Ensure each string in `solution_array` and the `metadata_dict` strictly adheres to all rules mentioned above:
{{
    "question": "<Your generated financial question, as a string>",
    "solution_array": <The Solution Steps Array as defined and detailed above, this is a list of strings>,
    "metadata_dict": <The Metadata Dictionary as defined and detailed above>,
    "expected_answer": "<The final single numeric answer with appropriate unit, matching the one in the 'Final Conclusion' step>"
}}
"""
    response_text = call_llm(llm_a1, prompt, A1_MODEL_ID, max_retries=2, retry_delay_seconds=10) # Increased retries for complex generation
    if not response_text:
        return None

    data = extract_json_from_response(response_text)

    if data and \
       'question' in data and \
       'solution_array' in data and isinstance(data['solution_array'], list) and \
       'metadata_dict' in data and isinstance(data['metadata_dict'], dict) and \
       'expected_answer' in data:
        print("A1 Generation Successful (JSON Parsed with required components).")
        # Basic validation of component types
        if not all(isinstance(step, str) for step in data['solution_array']):
            print("Error: Not all items in 'solution_array' are strings.")
            return None
        # Add more detailed validation of solution_array and metadata_dict content against QA rules here if needed,
        # though the prompt is very explicit.
        return data
    else:
        print(f"Error: A1 generation response parsing failed or missing required keys/correct types.")
        if data:
            print(f"Missing/invalid keys from {{'question', 'solution_array' (list), 'metadata_dict' (dict), 'expected_answer'}}. Found: { {k: type(v).__name__ for k,v in data.items()} }")
        else:
            print(f"Raw response from A1 that failed parsing or extraction: {response_text[:500]}...")
        return None


def attempt_solution_independent(llm_a2: LLM, excerpt_text: str, question: str) -> Optional[Dict[str, Any]]:
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
    response_text = call_llm(llm_a2, prompt, A2_MODEL_ID, max_retries=3, retry_delay_seconds=7)
    if not response_text:
        return None
    data = extract_json_from_response(response_text)
    if data and 'answer' in data and 'reasoning' in data:
        print("A2 Solution Attempt Successful (JSON Parsed).")
        return data
    else:
        print(f"Error: A2 solving response parsing failed or missing required keys.")
        if data: print(f"Missing keys from {{'answer', 'reasoning'}}. Found: {list(data.keys())}")
        else: print(f"Raw response from A2: {response_text[:500]}...")
        return None

def evaluate_independent_solution(llm_a1: LLM, question_data: dict, a2_response: dict) -> Optional[Dict[str, Any]]:
    """
    Agent A1: Evaluates A2's independent solution attempt.
    'question_data' now contains the fully formatted 'A1_Internal_Approach' string.
    """
    def _to_float(value_str):
        try:
            cleaned_str = str(value_str).replace('$', '').replace(',', '').replace('%', '').strip()
            return float(cleaned_str)
        except (ValueError, TypeError): return None

    a1_expected_answer_str = question_data.get('A1_Expected_Answer', 'N/A') # This is the direct answer string
    a2_answer_str = a2_response.get('answer', 'N/A')

    a1_answer_float = _to_float(a1_expected_answer_str)
    a2_answer_float = _to_float(a2_answer_str)

    numerical_match_flag = "Could not compare numerically"
    if a1_answer_float is not None and a2_answer_float is not None:
        is_close = abs(a1_answer_float - a2_answer_float) < 0.02 * abs(a1_answer_float) if a1_answer_float != 0 else abs(a2_answer_float) < 0.001 # Adjusted for zero case
        is_close = is_close or abs(a1_answer_float - a2_answer_float) < 0.02 # Absolute tolerance
        numerical_match_flag = "Yes" if is_close else "No"
    elif str(a1_expected_answer_str).strip().lower() == str(a2_answer_str).strip().lower():
        numerical_match_flag = "Yes (Exact String Match)"
    
    # The 'internal_approach' in question_data IS the fully formatted "Refer to file..." string
    a1_full_internal_approach_str = question_data.get('A1_Internal_Approach', 'N/A')

    prompt = f"""
You are the expert financial analyst (Agent A1) who originally created a question and a detailed internal solution based *only* on a specific financial document excerpt. You are now evaluating an attempt by another agent (Agent A2) who had to devise their *own* method to solve the question using *only* the same excerpt.

Your Original Question:
"{question_data.get('Question', 'N/A')}"

Your Detailed Internal Solution (this is the ground truth, including how to find data using 'Page X' markers from the processed excerpt, and the required formatting for steps):
{a1_full_internal_approach_str}

Your Calculated Expected Answer (derived strictly from the excerpt and your internal solution):
"{a1_expected_answer_str}"
---
Agent A2's Attempt:

A2's Provided Answer:
"{a2_answer_str}"

A2's Provided Reasoning/Method:
"{a2_response.get('reasoning', 'N/A')}"
---
Evaluation Task:
Perform the following evaluation based *only* on the information provided above and the assumption that both agents were restricted to the *same original excerpt*.

1.  **Numerical Correctness:** Compare A2's Provided Answer to Your Calculated Expected Answer.
    * My Expected Answer: {a1_expected_answer_str}
    * A2's Answer: {a2_answer_str}
    * Are they numerically equivalent, allowing for minor rounding differences (e.g., 0.724 vs 0.72)? (Pre-calculated Numerical Match: {numerical_match_flag})

2.  **Method Validity & Reasoning:** Analyze A2's Provided Reasoning/Method.
    * Does A2's reasoning demonstrate understanding of the question's core financial concept?
    * Did A2 devise a *logically valid* method to arrive at *an* answer using data plausibly found in the original excerpt? (Focus on logical flow).
    * Does A2's reasoning clearly state which data points it *claims* to have used from the excerpt?
    * Does the calculation described in A2's reasoning logically lead to A2's provided answer? (Check A2's internal consistency).
    * Critically: Is A2's method and reasoning plausible *given the constraint that it must rely solely on the original excerpt*? Does it seem to invent data or use external knowledge?

3.  **Overall Result:** Classify the attempt as 'Success' or 'Failure'.
    * 'Success' requires BOTH the numerical answer to be essentially correct (as per Numerical Match Flag starting with 'Yes') AND the reasoning/method (A2's Reasoning/Method) to be sound, logically valid, and demonstrably based *only* on information plausibly available in the original excerpt.
    * 'Failure' occurs if the numerical answer is incorrect (Numerical Match Flag = 'No') OR if A2's reasoning/method is flawed.

4.  **Failure Analysis (If Applicable):** If the result is 'Failure', provide a *brief* and specific 'analysis' explaining the primary reason(s) for the failure.

Output Format:
Respond *only* with a single, valid JSON object.
- Include the key 'result' with a value of either 'Success' or 'Failure'.
- ONLY if the 'result' is 'Failure', ALSO include the key 'analysis' with your brief explanation.

Example Success JSON Output:
{{
  "result": "Success"
}}

Example Failure JSON Output:
{{
  "result": "Failure",
  "analysis": "A2's numerical answer was incorrect. A2's reasoning used 'Total Assets' instead of 'Net Revenue' for the calculation, which is a conceptual error for this specific question type."
}}
"""
    response_text = call_llm(llm_a1, prompt, f"{A1_MODEL_ID} (Evaluation)", max_retries=1)
    if not response_text: return None
    data = extract_json_from_response(response_text)
    if data and 'result' in data:
        if data['result'] == 'Failure' and 'analysis' not in data:
            print(f"Warning: A1 evaluation is 'Failure' but missing 'analysis'. Adding placeholder.")
            data['analysis'] = "Evaluation marked Failure, but A1 did not provide specific analysis."
        elif data['result'] not in ['Success', 'Failure']:
            print(f"Error: A1 evaluation 'result' has invalid value: {data['result']}")
            return None
        print(f"A1 Evaluation Successful. Result: {data['result']}")
        return data
    else:
        print(f"Error: A1 evaluation parsing failed or missing 'result' key.")
        if data: print(f"Missing key 'result'. Found: {list(data.keys())}")
        else: print(f"Raw response from A1 (Evaluation): {response_text[:500]}...")
        return None

# --- Orchestrator Function ---
def run_framework(pdf_path: Path, llm_a1: LLM, llm_a2: LLM, target_failures: int = 2, max_iterations: int = 5) -> List[Dict[str, Any]]:
    failure_count = 0
    failure_log_list = []
    iteration = 0
    previous_questions: Set[str] = set()

    pdf_file_str = str(pdf_path)
    print(f"\nLoading financial excerpt from: {pdf_file_str}")
    if not pdf_path.is_file():
        print(f"Error: PDF file not found at {pdf_file_str}"); return []
    try:
        reader = PDFReader()
        documents: List[Document] = reader.load_data(file=pdf_path)
        if not documents: print(f"Error: No content extracted from {pdf_file_str}"); return []
        page_texts = []
        for i, doc in enumerate(documents):
            page_label = doc.metadata.get('page_label', str(i + 1)) # Use simple page number if no label
            file_name = doc.metadata.get('file_name', pdf_path.name)
            header = f"--- START OF Page {page_label} FROM {file_name} ---"
            footer = f"--- END OF Page {page_label} FROM {file_name} ---"
            page_texts.append(f"{header}\n{doc.text}\n{footer}")
        excerpt_text = "\n\n".join(page_texts)
        print(f"Successfully loaded text from {len(documents)} page(s) of PDF (total length: {len(excerpt_text)} chars).")
        if len(excerpt_text) < 100: print("Warning: Extracted text seems very short.")
    except Exception as e:
        print(f"Error reading or parsing PDF file {pdf_file_str}: {e}"); return []

    print(f"\nStarting framework run. Target Failures: {target_failures}, Max Iterations: {max_iterations}")
    print(f"Logging iterations to: {LOG_FILENAME}")
    initialize_log_file(LOG_FILENAME, LOG_FIELDNAMES)

    while failure_count < target_failures and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*15} Iteration {iteration} {'='*15}")
        time.sleep(1)

        q_data = None
        a2_resp = None
        eval_result = None
        current_log_data: Dict[str, Any] = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Iteration': iteration,
            'A1_Model': A1_MODEL_ID, 'A2_Model': A2_MODEL_ID,
            'Question': 'GENERATION FAILED', 'A1_Internal_Approach': '', 'A1_Expected_Answer': '',
            'A2_Provided_Answer': 'GENERATION FAILED', 'A2_Provided_Reasoning': '',
            'Evaluation_Result': 'N/A', 'Failure_Analysis': 'A1 Generation Failed'
        }

        print("\nStep 1: A1 generating question and internal solution components...")
        q_data = generate_question_internal_solution(llm_a1, excerpt_text, previous_questions)

        if q_data:
            generated_question = q_data.get('question', 'N/A')
            expected_answer = q_data.get('expected_answer', 'N/A')
            solution_array = q_data.get('solution_array', []) # This is a list of strings
            metadata_dict = q_data.get('metadata_dict', {})   # This is a dictionary
            
            print(f"A1 Generated Question: {generated_question}")
            previous_questions.add(generated_question)

            # Construct the annotations dictionary first
            annotations_content = {
                "question": generated_question,
                "answer": expected_answer,
                "solution": solution_array, # Keep as list of strings for now
                "metadata": metadata_dict
            }

            # Now, convert this dictionary to a pretty-printed JSON string
            # This will handle escaping of \n within the solution_array strings correctly for JSON
            annotations_as_json_string = json.dumps(annotations_content, indent=2)
            
            # Construct the final A1_Internal_Approach string
            # The 'id' should be the filename, and 'annotations' will be the JSON string we just created
            a1_internal_approach_str = f"""Refer to file: {Path(PDF_FILENAME).name}
All pages from the Corporate Annual Report
{{
  "id": "{Path(PDF_FILENAME).name}",
  "annotations": {annotations_as_json_string}
}}"""
            # Note: The annotations_as_json_string is directly embedded.
            # If the output format *requires* the "annotations" value to be a string
            # that *itself* contains a valid JSON object (like `{"annotations": "{...}"}`),
            # then `json.dumps(annotations_as_json_string)` would be needed.
            # However, the example implies `annotations` is an object key.

            current_log_data.update({
                'Question': generated_question,
                'A1_Internal_Approach': a1_internal_approach_str,
                'A1_Expected_Answer': expected_answer,
                'A2_Provided_Answer': 'ATTEMPT SKIPPED', 'A2_Provided_Reasoning': '',
                'Evaluation_Result': 'N/A', 'Failure_Analysis': ''
            })
        else:
            print(f"ERROR: Iteration {iteration} - A1 generation failed.")
            write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)
            continue

        print("\nStep 2: A2 attempting independent solution...")
        a2_resp = attempt_solution_independent(llm_a2, excerpt_text, current_log_data['Question'])
        if a2_resp:
            print(f"A2 Answer: {a2_resp.get('answer', 'N/A')}")
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
            continue

        print("\nStep 3: A1 evaluating A2's independent solution...")
        # Pass relevant parts of current_log_data for evaluation context
        eval_context = {
            "Question": current_log_data['Question'],
            "A1_Internal_Approach": current_log_data['A1_Internal_Approach'], # This is the full string
            "A1_Expected_Answer": current_log_data['A1_Expected_Answer']
        }
        eval_result = evaluate_independent_solution(llm_a1, eval_context, a2_resp)
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

        write_log_entry(LOG_FILENAME, LOG_FIELDNAMES, current_log_data)

        if eval_result and eval_result.get('result') == 'Failure':
            failure_count += 1
            analysis = eval_result.get('analysis', 'No analysis provided.')
            print(f"***** FAILURE LOGGED ({failure_count}/{target_failures}) *****")
            # The logged A1_Internal_Approach is already the full string
            failure_log_list.append({
                "iteration": iteration,
                "question": current_log_data['Question'],
                "a1_internal_approach_full_string": current_log_data['A1_Internal_Approach'],
                "a1_expected_answer": current_log_data['A1_Expected_Answer'],
                "a2_provided_answer": current_log_data['A2_Provided_Answer'],
                "a2_provided_reasoning": current_log_data['A2_Provided_Reasoning'],
                "failure_analysis": analysis,
                "a1_model": A1_MODEL_ID, "a2_model": A2_MODEL_ID
            })
        elif eval_result and eval_result.get('result') == 'Success':
            print("----- SUCCESS ----- (A2's independent approach was valid)")
        print(f"{'='*40}")

    if iteration >= max_iterations and failure_count < target_failures:
        print(f"\nWarning: Reached max iterations ({max_iterations}) before target failures ({target_failures}).")
    elif failure_count >= target_failures:
        print(f"\nTarget number of failures ({target_failures}) reached.")

    print(f"\nFramework run finished. Detailed log available in: {LOG_FILENAME}")
    return failure_log_list


# --- Main Execution Block ---
if __name__ == "__main__":
    if not google_api_key: exit()

    print("\n" + "="*50 + "\n Starting Multi-Agent Framework Run with LlamaIndex \n" + "="*50)

    try:
        print(f"Instantiating A1 LLM: {A1_MODEL_ID} with temp {TEMPERATUREA1}")
        llm_a1 = GoogleGenAI(model_name=A1_MODEL_ID, api_key=google_api_key, temperature=TEMPERATUREA1)
        print(f"Instantiating A2 LLM: {A2_MODEL_ID} with temp {TEMPERATUREA2}")
        llm_a2 = GoogleGenAI(model_name=A2_MODEL_ID, api_key=google_api_key, temperature=TEMPERATUREA2)
        print("LLM instances created successfully.")
    except Exception as e:
        print(f"FATAL: Failed to instantiate LLMs: {e}")
        if "model not found" in str(e).lower() or "could not find model" in str(e).lower():
            print(f"Verify model IDs ('{A1_MODEL_ID}', '{A2_MODEL_ID}') and API key access.")
        exit()

    TARGET_FAILURES = 2
    MAX_ITERATIONS = 5 
    pdf_file_path = Path(PDF_FILENAME)

    if not pdf_file_path.is_file():
        print(f"\nFATAL ERROR: PDF file '{PDF_FILENAME}' not found in ({Path.cwd()}).")
    else:
        failure_report = run_framework(
            pdf_path=pdf_file_path, llm_a1=llm_a1, llm_a2=llm_a2,
            target_failures=TARGET_FAILURES, max_iterations=MAX_ITERATIONS
        )
        print("\n" + "="*50 + f"\n Failure Summary Report (Logged {len(failure_report)} failures) \n Complete iteration log in: {LOG_FILENAME} \n" + "="*50)
        if not failure_report:
            print("No failures meeting criteria were logged within the iteration limit.")
        else:
            report_filename = f"{PDF_BASE_NAME}_failure_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(report_filename, 'w', encoding='utf-8') as f:
                    json.dump(failure_report, f, indent=2, ensure_ascii=False) # indent=2 for readability
                print(f"Failure summary saved to: {report_filename}")
            except Exception as e:
                print(f"Error saving failure summary to JSON: {e}")
            for i, failure in enumerate(failure_report):
                print(f"\n--- Failure Summary #{i+1} (Iteration: {failure['iteration']}) ---")
                print(f"  Models: A1={failure['a1_model']}, A2={failure['a2_model']}")
                print(f"  Question: {failure['question']}")
                print(f"  A1 Expected: {failure['a1_expected_answer']}, A2 Provided: {failure['a2_provided_answer']}")
                print(f"  Failure Analysis: {failure['failure_analysis']}")
                # Optionally print parts of a1_internal_approach_full_string if not too long
                # print(f"  A1 Internal Approach (excerpt): {failure['a1_internal_approach_full_string'][:300]}...")
    print("\n" + "="*50 + "\n Framework Simulation Complete \n" + "="*50)