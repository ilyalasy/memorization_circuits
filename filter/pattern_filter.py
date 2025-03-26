import re
import os
from difflib import SequenceMatcher

def is_sequential_numbers(text, completion):
    """Detect if completion is just a continuation of sequential numbers."""
    # Extract numbers from text and completion
    text_numbers = [int(n) for n in re.findall(r'\b(\d+)\b', text)]
    completion_numbers = [int(n) for n in re.findall(r'\b(\d+)\b', completion)]
    
    if not text_numbers or not completion_numbers:
        return False
    
    # Check if the completion numbers are sequential to the text numbers
    last_text_num = text_numbers[-1] if text_numbers else 0
    if all(completion_numbers[i] == last_text_num + i + 1 for i in range(len(completion_numbers))):
        return True
    
    # Check if numbers within completion are sequential
    if len(completion_numbers) > 1:
        is_sequential = True
        for i in range(1, len(completion_numbers)):
            if completion_numbers[i] != completion_numbers[i-1] + 1:
                is_sequential = False
                break
        if is_sequential:
            return True
    
    return False

def is_line_by_line_sequential(text, completion):
    """Check if the text and completion have a line-by-line number incrementing pattern."""
    text_lines = text.strip().split('\n')
    completion_lines = completion.strip().split('\n')
    
    # Need sufficient lines to detect a pattern
    if len(text_lines) < 2 or len(completion_lines) < 2:
        return False
    
    # Extract numbers from each line
    text_line_numbers = []
    for line in text_lines:
        numbers = [int(n) for n in re.findall(r'\b(\d+)\b', line)]
        if numbers:
            text_line_numbers.append(numbers)
    
    completion_line_numbers = []
    for line in completion_lines:
        numbers = [int(n) for n in re.findall(r'\b(\d+)\b', line)]
        if numbers:
            completion_line_numbers.append(numbers)
    
    # Need sufficient numbered lines
    if len(text_line_numbers) < 2 or len(completion_line_numbers) < 2:
        return False
    
    # Check if each line's number is incrementing by a consistent amount
    text_increments = []
    for i in range(1, len(text_line_numbers)):
        if len(text_line_numbers[i]) == len(text_line_numbers[i-1]):
            increments = [text_line_numbers[i][j] - text_line_numbers[i-1][j] 
                         for j in range(len(text_line_numbers[i]))]
            text_increments.append(increments)
    
    completion_increments = []
    for i in range(1, len(completion_line_numbers)):
        if len(completion_line_numbers[i]) == len(completion_line_numbers[i-1]):
            increments = [completion_line_numbers[i][j] - completion_line_numbers[i-1][j] 
                         for j in range(len(completion_line_numbers[i]))]
            completion_increments.append(increments)
    
    # If we have consistent increments in both text and completion
    all_text_increments = []
    if text_increments and completion_increments:
        # Check if all increments in text are the same
        all_text_increments = [inc for sublist in text_increments for inc in sublist]
        all_completion_increments = [inc for sublist in completion_increments for inc in sublist]
        
        if all_text_increments and all(x == all_text_increments[0] for x in all_text_increments) and \
           all_completion_increments and all(x == all_completion_increments[0] for x in all_completion_increments) and \
           all_text_increments[0] == all_completion_increments[0]:
            return True
    
    # Check if completion continues from text
    if text_line_numbers and completion_line_numbers:
        last_text_numbers = text_line_numbers[-1]
        first_completion_numbers = completion_line_numbers[0]
        
        if len(last_text_numbers) == len(first_completion_numbers):
            expected_first_completion = [last_text_numbers[i] + (all_text_increments[0] if all_text_increments else 1) 
                                       for i in range(len(last_text_numbers))]
            if expected_first_completion == first_completion_numbers:
                return True
    
    return False

def is_repeated_text(text, completion, similarity_threshold=0.8):
    """Detect if completion is just repeating text from the input."""
    # If the completion is very similar to the text
    if not text or not completion:
        return False
    
    # Simple repetition check
    if text.strip() == completion.strip():
        return True
    
    # Check if completion repeats a pattern from text
    # Take segments from text and see if they appear in completion
    words_text = text.split()
    words_completion = completion.split()
    
    if len(words_text) >= 3 and len(words_completion) >= 3:
        # Check for repeated phrases (at least 3 words)
        for i in range(len(words_text) - 2):
            phrase = ' '.join(words_text[i:i+3])
            if phrase in ' '.join(words_completion):
                return True
    
    # Check for line-by-line repetition patterns
    text_lines = text.strip().split('\n')
    completion_lines = completion.strip().split('\n')
    
    if len(text_lines) > 0 and len(completion_lines) > 0:
        # Get the pattern of the last few lines of text
        pattern_lines = text_lines[-min(3, len(text_lines)):]
        pattern = '\n'.join(pattern_lines)
        
        # See if this pattern repeats in completion
        completion_text = '\n'.join(completion_lines)
        if pattern in completion_text:
            return True
    
    # Check for character-level patterns 
    # (e.g., repeating dots, repeating characters like in row_14344)
    if len(text) > 3 and len(completion) > 3:
        # If text contains only one unique character repeated (ignoring whitespace)
        text_chars = set(ch for ch in text if not ch.isspace())
        completion_chars = set(ch for ch in completion if not ch.isspace())
        
        if len(text_chars) == 1 and len(completion_chars) == 1 and text_chars == completion_chars:
            return True
    
    return False

def is_simple_structure_continuation(text, completion):
    """Detect if completion is just continuing a simple structure like HTML, tables, etc."""
    # Common structural patterns
    patterns = [
        # HTML tags continuations
        r'<[a-zA-Z]+>.*?</[a-zA-Z]+>',
        # Table rows
        r'\|-\s*\n\|\s',
        # List items with consistent formatting
        r'^\s*\d+\.\s',
        r'^\s*•\s',
        r'^\s*\*\s',
        # File path patterns
        r'[A-Za-z]+/[A-Za-z]+\d+',
        # Line numbers with similar formatting
        r'^\s+\d+\.\s',
        # License text patterns
        r'Licensed under the Apache License',
        # Category patterns
        r'^Category:',
        # Links and references patterns
        r'!\[\]\(.*?\)',
        r'\[\]\(.*?\)',
        # Copyright statements
        r'Copyright \d{4}',
        # Biblical verse references
        r'[A-Za-z]+ \d+:\d+',
        # Character references 
        r'&#\d+;',
        # CSS class attributes
        r'class="[^"]+"',
        # Bibliography citation patterns
        r'\[@bib\d+\]',
        # DOCTYPE declarations
        r'<!DOCTYPE',
        # XML namespaces
        r'xmlns(:[a-zA-Z]+)?="[^"]+"',
    ]
    
    # Check if both text and completion match the same pattern
    for pattern in patterns:
        text_matches = re.findall(pattern, text, re.MULTILINE)
        completion_matches = re.findall(pattern, completion, re.MULTILINE)
        
        if text_matches and completion_matches:
            # If the pattern appears with similar frequency, it might be a simple continuation
            if len(text_matches) / max(1, len(text.split('\n'))) > 0.3 and len(completion_matches) / max(1, len(completion.split('\n'))) > 0.3:
                return True
    
    # Check for similar line structures
    text_lines = text.strip().split('\n')
    completion_lines = completion.strip().split('\n')
    
    if len(text_lines) >= 2 and len(completion_lines) >= 2:
        # Get structure of first few lines (ignoring content)
        text_structure = []
        for line in text_lines[:min(5, len(text_lines))]:
            # Replace actual content with placeholders, keeping structure
            structure = re.sub(r'[a-zA-Z0-9]+', 'X', line)
            text_structure.append(structure)
        
        completion_structure = []
        for line in completion_lines[:min(5, len(completion_lines))]:
            structure = re.sub(r'[a-zA-Z0-9]+', 'X', line)
            completion_structure.append(structure)
        
        # If structures are similar
        if text_structure and completion_structure:
            similarity = SequenceMatcher(None, str(text_structure), str(completion_structure)).ratio()
            if similarity > 0.8:
                return True
    
    # Check for non-Latin character patterns (CJK, Cyrillic, etc.)
    if any(ord(c) > 127 for c in text) and any(ord(c) > 127 for c in completion):
        # If both text and completion contain non-Latin characters
        non_latin_text = ''.join(c for c in text if ord(c) > 127)
        non_latin_completion = ''.join(c for c in completion if ord(c) > 127)
        
        # If they contain similar script types (CJK, Cyrillic, etc.)
        if non_latin_text and non_latin_completion:
            text_script_counts = {}
            completion_script_counts = {}
            
            # Count character scripts
            for c in non_latin_text:
                script = get_script_category(c)
                text_script_counts[script] = text_script_counts.get(script, 0) + 1
            
            for c in non_latin_completion:
                script = get_script_category(c)
                completion_script_counts[script] = completion_script_counts.get(script, 0) + 1
            
            # If dominant scripts are the same
            if text_script_counts and completion_script_counts:
                dominant_text_script = max(text_script_counts, key=text_script_counts.get)
                dominant_completion_script = max(completion_script_counts, key=completion_script_counts.get)
                
                if dominant_text_script == dominant_completion_script:
                    # If the non-Latin text follows a similar pattern
                    # Check for repetitive patterns
                    if is_repetitive_pattern(non_latin_text) and is_repetitive_pattern(non_latin_completion):
                        return True
    
    return False

def get_script_category(char):
    """Return a general script category for a character."""
    code_point = ord(char)
    
    # CJK Unified Ideographs
    if 0x4E00 <= code_point <= 0x9FFF:
        return 'CJK'
    # Hiragana
    elif 0x3040 <= code_point <= 0x309F:
        return 'Hiragana'
    # Katakana
    elif 0x30A0 <= code_point <= 0x30FF:
        return 'Katakana'
    # Cyrillic
    elif 0x0400 <= code_point <= 0x04FF:
        return 'Cyrillic'
    # Arabic
    elif 0x0600 <= code_point <= 0x06FF:
        return 'Arabic'
    # Devanagari
    elif 0x0900 <= code_point <= 0x097F:
        return 'Devanagari'
    # Thai
    elif 0x0E00 <= code_point <= 0x0E7F:
        return 'Thai'
    # Korean Hangul
    elif 0xAC00 <= code_point <= 0xD7A3:
        return 'Hangul'
    # Default
    return 'Other'

def is_repetitive_pattern(text, min_segment_length=2):
    """Check if text contains repetitive patterns."""
    if len(text) < min_segment_length * 2:
        return False
    
    # Check for repeating segments
    for segment_length in range(min_segment_length, len(text) // 2 + 1):
        segment = text[:segment_length]
        # See if this segment repeats
        repeats = text.count(segment)
        if repeats > 1 and repeats * len(segment) > len(text) / 2:
            return True
    
    return False

def is_simple_pattern(text, completion):
    """Combine all pattern detection methods."""
    return (
        is_sequential_numbers(text, completion) or
        is_line_by_line_sequential(text, completion) or
        is_repeated_text(text, completion) or
        is_simple_structure_continuation(text, completion)
    )

def extract_examples_from_source(source_file):
    """Extract examples directly from the source file."""
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If the file is from additional_data with proper formatting
    if '<file_contents>' in content:
        # Extract the examples.txt content
        match = re.search(r'```data/mem/examples.txt\n(.*?)```', content, re.DOTALL)
        if match:
            examples_content = match.group(1)
            return examples_content
    
    # Otherwise assume the file directly contains the examples
    return content

def filter_samples(input_content, output_dir="filtered_output"):
    """Filter out simple patterns from the examples content."""
    # Regular expression to extract row examples
    row_pattern = r'<row_(\d+)>(.*?)<completion>(.*?)</row_\1>'
    matches = re.findall(row_pattern, input_content, re.DOTALL)
    
    total_samples = len(matches)
    filtered_samples = 0
    kept_samples = []
    filtered_row_ids = []
    
    for row_id, text, completion in matches:
        if not is_simple_pattern(text, completion):
            kept_samples.append((row_id, text, completion))
        else:
            filtered_samples += 1
            filtered_row_ids.append(row_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write kept samples to output file
    with open(os.path.join(output_dir, 'filtered_examples.txt'), 'w', encoding='utf-8') as f:
        for row_id, text, completion in kept_samples:
            f.write(f"<row_{row_id}>\n{text}\n<completion>{completion}</row_{row_id}>\n\n")
    
    # Write filtered row IDs to a separate file for reference
    with open(os.path.join(output_dir, 'filtered_row_ids.txt'), 'w', encoding='utf-8') as f:
        f.write("Filtered row IDs:\n")
        f.write('\n'.join(filtered_row_ids))
    
    print(f"Total samples: {total_samples}")
    print(f"Filtered samples: {filtered_samples}")
    print(f"Remaining samples: {len(kept_samples)}")
    print(f"Filtered percentage: {filtered_samples/total_samples*100:.2f}%")
    print(f"Results written to {os.path.join(output_dir, 'filtered_examples.txt')}")
    print(f"Filtered row IDs written to {os.path.join(output_dir, 'filtered_row_ids.txt')}")
    
    return {
        'total': total_samples,
        'filtered': filtered_samples,
        'kept': len(kept_samples),
        'filtered_ids': filtered_row_ids
    }

if __name__ == "__main__":
    # Try both potential file locations
    for source_path in ["data/mem/examples.txt", "examples.txt", "additional_data.txt"]:
        if os.path.exists(source_path):
            print(f"Processing examples from: {source_path}")
            examples_content = extract_examples_from_source(source_path)
            filter_samples(examples_content)
            break
    else:
        print("No examples file found. Please provide the file location directly.")
        # If examples file is not found, look for attached data in the script's directory
        for file in os.listdir('.'):
            if 'additional_data' in file or 'attached' in file or 'examples' in file:
                print(f"Found potential examples in file: {file}")
                examples_content = extract_examples_from_source(file)
                filter_samples(examples_content)
                break 