import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import PyPDF2
from openai import OpenAI
import statistics
from typing import List, Tuple, Optional
import argparse

# --- Grading API Configuration ---
GRADING_API_KEY = "your-openai-key"
GRADING_BASE_URL = "https://api.openai.com/v1"
GRADING_MODEL = "gpt-4.1"

# --- JSON Extraction API Configuration ---
EXTRACTION_API_KEY = "your-openai-key"
EXTRACTION_BASE_URL = "https://api.openai.com/v1"
EXTRACTION_MODEL = "gpt-4.1-mini"

# --- Model Configuration ---
TEMPERATURE = 0.35
GRADING_DELAY = 60  # delay between grading API calls (seconds)
SCORE_TOLERANCE = 5  # Within 5 marks range

# Initialize OpenAI clients
grading_client = OpenAI(
    api_key=GRADING_API_KEY,
    base_url=GRADING_BASE_URL
)

extraction_client = OpenAI(
    api_key=EXTRACTION_API_KEY,
    base_url=EXTRACTION_BASE_URL
)

# Input and output directories
INPUT_FOLDER = "./input"
OUTPUT_FILE = "manuscript_grading_results.xlsx"
DEFAULT_OUTPUT_FILE = "manuscript_grading_results.xlsx"
PHRASE_FILE = os.path.join(os.path.dirname(__file__), "remove.txt")

def load_phrases_to_remove(phrase_file_path: str) -> List[str]:
    """Load set phrases to remove from a file (one per line, ignores blanks and strips whitespace)."""
    if not os.path.isfile(phrase_file_path):
        print(f"Phrase file {phrase_file_path} not found, proceeding with no phrases.")
        return []
    with open(phrase_file_path, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    return phrases

def remove_dois_and_phrases(text: str, phrases: List[str]) -> str:
    """Remove all DOIs and set phrases (case-insensitive, phrase as substring) from the text."""
    # DOI regex (matches both http(s)://doi.org/DOI and just bare DOIs)
    doi_pattern = re.compile(
        r"(?:\b(?:https?://)?(?:dx\.)?doi\.org/|\bdoi:\s*|\bDOI:\s*|\b)(10\.\d{4,9}/\S+?)(?=[\s\.,;]|$)",
        re.IGNORECASE
    )
    text = doi_pattern.sub("", text)

    # Remove each phrase, case-insensitive, match as substring
    for phrase in phrases:
        if not phrase:
            continue
        # Use re.escape in case phrases have special regex characters
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)

    # Remove any doubled-up whitespace/newlines caused by removal
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def get_system_prompt() -> str:
    """Create the system prompt with grading instructions"""
    return """You are tasked with grading manuscripts in the intersection of Chemistry, Solid State Physics and Material Science. Six prestige levels of well-known journals are listed below for reference, but you do not need to consider manuscript fit to specific journals; these are to convey the relative prestige of each level. For each manuscript, provide your educated guess—expressed as a percentage—for the chance it would be sent out for external review at each of the six journal levels. In making your estimates, consider overall quality, scope breadth, methodological novelty, interest to readership, workload, quality of writing, methodological rigour, and whether the results fully support the claims. Some manuscripts you grade may already be published articles, but please evaluate them as if they are new, without regard to where they were actually published. Note each lower level should have a equal or higher chance than the previous level.

*Level 1 - Very top journal*
Nature; Science.

*Level 2 - Well-regarded sister journals of the very top journals*
Nature Materials; Nature Chemistry; Nature Electronic; Nature Physics; Nature Energy; Nature Nanotechnology.

*Level 3 - Weaker sister journals of the very top journals or equivalent*
Nature Communications; Science Advances; Joule.

*Level 4 - Top broad field journals*
Journal of the American Chemical Society; Advanced Materials; Angewandte Chemie International Edition; Physical Review Letters; Physical Review X; PNAS; Chem.

*Level 5 - Good broad field journals and equivalent*
JACS Au; Advanced Functional Materials; Advanced Energy Materials; CCS Chemistry; Chemical Science; ACS Central Science; Advanced Science; Energy and Environmental Science; Physical Review B

*Level 6 - Good narrow field journals*
Journal of Materials Chemistry A; Small; Chemical Communications; Macromolecules; Advanced Electronic Materials; Journal of Physical Chemistry Letters; Journal of Physical Chemistry A; ACS Applied Materials & Interface.

## Output Format
Your output must be a JSON object with:
- Keys Level 1 through Level 6, each mapping to an integer percentage (0–100) indicating your estimate of the chance the manuscript is sent for external review at that level.
- A single key "justification" with a one-sentence rationale for your scoring.

Example output:
{
  "Level 1": 0,
  "Level 2": 10,
  "Level 3": 50,
  "Level 4": 80,
  "Level 5": 90,
  "Level 6": 100,
  "justification": "The methodology is solid, but the novelty and breadth do not meet the expectations for the highest-impact journals."
}"""

def get_user_prompt(manuscript_text: str) -> str:
    """Create the user prompt with manuscript text"""
    return f"Manuscript to grade:\n\n{manuscript_text}"

def get_json_extraction_prompt(raw_response: str) -> str:
    """Create the prompt for JSON extraction"""
    return f"""Please extract and format the following response into valid JSON format. The response should contain:
- Keys "Level 1" through "Level 6", each mapping to an integer percentage (0-100)
- A key "justification" with the rationale text

Here is the raw response to extract from:
{raw_response}

Please output only the valid JSON object, nothing else."""

def extract_json_with_model(raw_response: str) -> dict:
    """Use the extraction model to clean up and extract JSON from the raw response"""
    try:
        messages = [
            {"role": "system", "content": "You are a JSON extraction assistant. Extract and format responses into valid JSON objects."},
            {"role": "user", "content": get_json_extraction_prompt(raw_response)}
        ]
        
        response = extraction_client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=messages,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove any markdown formatting if present
        if result.startswith('```json'):
            result = result[7:]
        if result.endswith('```'):
            result = result[:-3]
        result = result.strip()
        
        # Parse the JSON
        return json.loads(result)
    except Exception as e:
        print(f"Error in JSON extraction: {str(e)}")
        raise

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Manuscript Grader: specify one or more input folders of .pdf files."
    )
    parser.add_argument(
        'input_folders',
        nargs='*',
        default=['./input'],
        help="One or more folders containing manuscripts (default: ./input)"
    )
    parser.add_argument(
        '--output-file',
        '-o',
        default=DEFAULT_OUTPUT_FILE,
        help=f"Excel filename to save results into (default: {DEFAULT_OUTPUT_FILE})"
    )
    return parser.parse_args()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file"""
    try:
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(text)
        
        extracted_text = "\n".join(text_content)
        
        # Basic cleanup - remove excessive whitespace
        extracted_text = re.sub(r'\n\s*\n', '\n\n', extracted_text)
        extracted_text = re.sub(r' +', ' ', extracted_text)
        
        return extracted_text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {str(e)}")
        print("Note: Make sure PyPDF2 is installed: pip install PyPDF2")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""

def grade_manuscript(manuscript_text: str):
    """Send manuscript to grading model, then extract JSON using extraction model"""
    try:
        # Step 1: Get grading response from grading model
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_user_prompt(manuscript_text)}
        ]
        
        response = grading_client.chat.completions.create(
            model=GRADING_MODEL,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        raw_result = response.choices[0].message.content
        time.sleep(1)  # Wait 1 second after grading API call
        
        # Step 2: Extract JSON using extraction model
        try:
            grading_data = extract_json_with_model(raw_result)
            time.sleep(1)  # Wait 1 second after extraction API call
            
            # Extract scores in order [Level 1, Level 2, ..., Level 6] and justification
            scores = []
            for i in range(1, 7):
                level_key = f"Level {i}"
                score = grading_data.get(level_key)
                # Ensure score is treated as a number, handle potential string values
                if score is not None:
                    scores.append(float(score))
                else:
                    scores.append(None)
            
            justification = grading_data.get("justification", "No justification provided")
            return scores, justification
            
        except Exception as e:
            print(f"Error in JSON extraction step: {str(e)}")
            print(f"Raw grading response: {raw_result}")
            
            # Fallback: try to parse the raw result directly
            try:
                grading_data = json.loads(raw_result)
                scores = []
                for i in range(1, 7):
                    level_key = f"Level {i}"
                    score = grading_data.get(level_key)
                    if score is not None:
                        scores.append(float(score))
                    else:
                        scores.append(None)
                justification = grading_data.get("justification", "No justification provided")
                return scores, justification
            except:
                return [None] * 6, "Error parsing response"
                
    except Exception as e:
        print(f"Error in grading: {str(e)}")
        time.sleep(1)  # Wait even on error
        return [None] * 6, "Error in API call"

def calculate_weighted_score(scores):
    """
    Calculate weighted average: level 1 weight 2, level 2 weight 2, others weight 1
    Args:
        scores: List of 6 scores [level1, level2, level3, level4, level5, level6]
    Returns:
        float: Weighted average score
    """
    weights = [2, 2, 1, 1, 1, 1]
    weighted_sum = 0
    total_weight = 0
    
    for i, score in enumerate(scores):
        if score is not None:
            weighted_sum += score * weights[i]
            total_weight += weights[i]
    
    if total_weight == 0:
        return 0
    return weighted_sum / total_weight

def robust_grade_manuscript(manuscript_text: str) -> Tuple[List[float], float, int, str, str]:
    """
    Grade manuscript with retry mechanism:
    1. Grade twice first
    2. If within 5 marks range, take average
    3. If not, grade third time and take average of closest pair
    
    Returns:
        Tuple of (final_scores, final_weighted_score, num_attempts, decision_reason, justification)
    """
    print("    Starting robust grading process...")
    
    # First grading
    print("      Grading attempt 1/3...")
    scores1, justification1 = grade_manuscript(manuscript_text)
    if scores1 is None or all(s is None for s in scores1):
        print("      ✗ First grading failed")
        return [0] * 6, 0, 1, "Failed: First grading failed", "Grading failed"
    
    weighted_score1 = calculate_weighted_score(scores1)
    print(f"      ✓ First grading complete: {weighted_score1:.1f}")
    
    # Wait before second grading
    print(f"      Waiting {GRADING_DELAY} seconds before next grading...")
    time.sleep(GRADING_DELAY)
    
    # Second grading
    print("      Grading attempt 2/3...")
    scores2, justification2 = grade_manuscript(manuscript_text)
    if scores2 is None or all(s is None for s in scores2):
        print("      ✗ Second grading failed, using first result")
        final_scores = [score if score is not None else 0 for score in scores1]
        return final_scores, weighted_score1, 2, "Used first result: Second grading failed", justification1
    
    weighted_score2 = calculate_weighted_score(scores2)
    print(f"      ✓ Second grading complete: {weighted_score2:.1f}")
    
    # Check if scores are within tolerance
    score_difference = abs(weighted_score1 - weighted_score2)
    print(f"      Score difference: {score_difference:.1f} (tolerance: {SCORE_TOLERANCE})")
    
    if score_difference <= SCORE_TOLERANCE:
        # Scores are close enough, take average
        print("      ✓ Scores within tolerance, taking average")
        final_scores = [(s1 + s2) / 2 for s1, s2 in zip(scores1, scores2)]
        final_weighted_score = (weighted_score1 + weighted_score2) / 2
        combined_justification = f"Average of two close scores. First: {justification1} Second: {justification2}"
        return final_scores, final_weighted_score, 2, "Average of two close scores", combined_justification
    else:
        # Scores differ too much, need third grading
        print("      ✗ Scores differ too much, attempting third grading...")
        
        # Wait before third grading
        print(f"      Waiting {GRADING_DELAY} seconds before third grading...")
        time.sleep(GRADING_DELAY)
        
        # Third grading
        print("      Grading attempt 3/3...")
        scores3, justification3 = grade_manuscript(manuscript_text)
        if scores3 is None or all(s is None for s in scores3):
            print("      ✗ Third grading failed, using average of first two")
            final_scores = [(s1 + s2) / 2 for s1, s2 in zip(scores1, scores2)]
            final_weighted_score = (weighted_score1 + weighted_score2) / 2
            combined_justification = f"Average of first two (third failed). First: {justification1} Second: {justification2}"
            return final_scores, final_weighted_score, 3, "Average of first two: Third grading failed", combined_justification
        
        weighted_score3 = calculate_weighted_score(scores3)
        print(f"      ✓ Third grading complete: {weighted_score3:.1f}")
        
        # Find the closest pair among the three scores
        diff_1_2 = abs(weighted_score1 - weighted_score2)
        diff_1_3 = abs(weighted_score1 - weighted_score3)
        diff_2_3 = abs(weighted_score2 - weighted_score3)
        
        print(f"      Score differences: 1-2: {diff_1_2:.1f}, 1-3: {diff_1_3:.1f}, 2-3: {diff_2_3:.1f}")
        
        if diff_1_2 <= diff_1_3 and diff_1_2 <= diff_2_3:
            # Scores 1 and 2 are closest
            print("      ✓ Taking average of first and second scores (closest pair)")
            final_scores = [(s1 + s2) / 2 for s1, s2 in zip(scores1, scores2)]
            final_weighted_score = (weighted_score1 + weighted_score2) / 2
            combined_justification = f"Average of closest pair (1st & 2nd). First: {justification1} Second: {justification2}"
            return final_scores, final_weighted_score, 3, "Average of closest pair (1st & 2nd)", combined_justification
        elif diff_1_3 <= diff_2_3:
            # Scores 1 and 3 are closest
            print("      ✓ Taking average of first and third scores (closest pair)")
            final_scores = [(s1 + s3) / 2 for s1, s3 in zip(scores1, scores3)]
            final_weighted_score = (weighted_score1 + weighted_score3) / 2
            combined_justification = f"Average of closest pair (1st & 3rd). First: {justification1} Third: {justification3}"
            return final_scores, final_weighted_score, 3, "Average of closest pair (1st & 3rd)", combined_justification
        else:
            # Scores 2 and 3 are closest
            print("      ✓ Taking average of second and third scores (closest pair)")
            final_scores = [(s2 + s3) / 2 for s2, s3 in zip(scores2, scores3)]
            final_weighted_score = (weighted_score2 + weighted_score3) / 2
            combined_justification = f"Average of closest pair (2nd & 3rd). Second: {justification2} Third: {justification3}"
            return final_scores, final_weighted_score, 3, "Average of closest pair (2nd & 3rd)", combined_justification


def process_single_manuscript(file_path, folder_name):
    """Process a single manuscript file and return results"""
    print(f"\nProcessing: {os.path.basename(file_path)} (from folder: {folder_name})")

    # Extract text from manuscript
    print(f"  Extracting text from manuscript...")
    manuscript_text = extract_text_from_file(file_path)
    if not manuscript_text:
        print(f"  ERROR: Failed to extract text from {file_path}")
        return None

    print(f"  Extracted {len(manuscript_text)} characters from manuscript")

    # === REMOVE DOIs AND SET PHRASES HERE ===
    phrases_to_remove = load_phrases_to_remove(PHRASE_FILE)
    cleaned_text = remove_dois_and_phrases(manuscript_text, phrases_to_remove)
    print(f"  Cleaned manuscript text: {len(cleaned_text)} characters after removal of DOIs/phrases")

    try:
        # Grade the manuscript (now use cleaned_text)
        final_scores, final_weighted_score, num_attempts, decision_reason, justification = robust_grade_manuscript(
            cleaned_text)

        print(f"  COMPLETED! Final Score: {final_weighted_score:.1f} (after {num_attempts} attempts)")
        print(f"    Decision: {decision_reason}")

        return {
            'filename': os.path.basename(file_path),
            'folder': folder_name,
            'level1': final_scores[0],
            'level2': final_scores[1],
            'level3': final_scores[2],
            'level4': final_scores[3],
            'level5': final_scores[4],
            'level6': final_scores[5],
            'final_score': final_weighted_score,
            'num_attempts': num_attempts,
            'decision_reason': decision_reason,
            'justification': justification
        }
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred: {e}")
        return None


def save_results_to_excel(results_list, output_file):
    """Save all results to Excel file"""
    try:
        # Create DataFrame
        df = pd.DataFrame(results_list)

        # Reorder columns
        column_order = ['filename', 'folder', 'level1', 'level2', 'level3', 'level4', 'level5', 'level6',
                        'final_score', 'num_attempts', 'decision_reason', 'justification']
        df = df[column_order]

        # Round numerical columns to 1 decimal place
        numerical_cols = ['level1', 'level2', 'level3', 'level4', 'level5', 'level6', 'final_score']
        for col in numerical_cols:
            df[col] = df[col].round(1)

        # Save to Excel
        df.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
        return False

def main():
    """Main function to process all manuscripts in specified input folders."""
    args = parse_args()
    input_folders = args.input_folders
    output_file = args.output_file

    print("=== Batch Manuscript Grader ===")
    print(f"Input folders: {', '.join(input_folders)}")
    print(f"Output file: {output_file}")
    print(f"Grading model: {GRADING_MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Grading delay: {GRADING_DELAY} seconds")
    print(f"Score tolerance: {SCORE_TOLERANCE} marks")
    print("Processing: Sequential with retry mechanism")
    print("Supported file types: .pdf")

    # Check if all input folders exist
    non_existing = [f for f in input_folders if not os.path.exists(f)]
    if non_existing:
        print(f"ERROR: The following input folders do not exist: {', '.join(non_existing)}")
        print("Please create them and place your .pdf files in them.")
        return

    # Collect all .pdf files from all input folders, and keep track of their folder name
    supported_extensions = {'.pdf'}
    manuscript_files = []
    for folder in input_folders:
        folder_files = [
            (os.path.join(folder, f), os.path.basename(os.path.abspath(folder)))
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in supported_extensions and not f.startswith('~')
        ]
        manuscript_files.extend(folder_files)

    if not manuscript_files:
        print(f"ERROR: No .pdf files found in any specified input folder!")
        return

    print(f"\nFound {len(manuscript_files)} manuscript files to process:")
    for file_path, folder_name in manuscript_files:
        ext = os.path.splitext(file_path)[1].upper()
        print(f"  - {os.path.basename(file_path)} [{ext}] (folder: {folder_name})")

    # Process each file sequentially
    results_list = []
    start_time = datetime.now()
    print(f"\n{'=' * 60}")
    print("Starting sequential processing...")

    for i, (file_path, folder_name) in enumerate(manuscript_files, start=1):
        print(f"\n[{i}/{len(manuscript_files)}] {os.path.basename(file_path)}")
        try:
            result = process_single_manuscript(file_path, folder_name)
            if result:
                results_list.append(result)
                print(f"✓ Successfully completed {os.path.basename(file_path)}")
            else:
                print(f"✗ Failed to process {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Exception processing {os.path.basename(file_path)}: {e}")

    # Save and summarise results
    if results_list:
        print(f"\n{'=' * 60}")
        print(f"Batch processing completed: {len(results_list)} succeeded, {len(manuscript_files) - len(results_list)} failed")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_with_ts = Path(output_file)
        output_with_ts = output_with_ts.with_name(f"{output_with_ts.stem}_{timestamp}{output_with_ts.suffix}")
        
        if save_results_to_excel(results_list, str(output_with_ts)):
            print(f"\nResults saved to {output_with_ts}")
        
        duration = datetime.now() - start_time
        print(f"Total processing time: {duration}")
        
        avg = duration.total_seconds() / len(results_list)
        print(f"Average time per manuscript: {avg:.1f} seconds")
        
        total_calls = sum(r['num_attempts'] for r in results_list)
        print(f"Total API calls made: {total_calls}")
        
        # Summary of grading attempts
        attempt_counts = {}
        for result in results_list:
            attempts = result['num_attempts']
            attempt_counts[attempts] = attempt_counts.get(attempts, 0) + 1
        
        print("\nGrading attempt summary:")
        for attempts in sorted(attempt_counts.keys()):
            count = attempt_counts[attempts]
            print(f"  {attempts} attempts: {count} manuscripts")
            
    else:
        print("ERROR: No files were successfully processed!")

if __name__ == "__main__":
    main()
