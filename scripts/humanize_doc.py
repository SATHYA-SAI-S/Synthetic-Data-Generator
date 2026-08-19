import argparse
import os
import subprocess

def save_to_file(text, filepath):
    ext = os.path.splitext(filepath)[1].lower()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if ext == '.docx':
        try:
            from docx import Document
        except ImportError:
            print("Warning: 'python-docx' is not installed. Run 'pip install python-docx'.")
            fallback_path = filepath.replace('.docx', '.md')
            print(f"Saving as markdown instead: {fallback_path}")
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return

        doc = Document()
        doc.add_paragraph(text)
        doc.save(filepath)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanize AI generated text using llmstrip.")
    parser.add_argument('--input', type=str, required=True, help='Path to a file containing raw text')
    parser.add_argument('--output', type=str, required=True, help='Path to save output file (.md or .docx)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        exit(1)
        
    print("Humanizing text with llmstrip...")
    
    # Run llmstrip on the input file
    llmstrip_exe = os.path.join(os.path.dirname(__file__), "llmstrip", "llmstrip.exe")
    if not os.path.exists(llmstrip_exe):
        print(f"Error: {llmstrip_exe} not found. Please download and extract llmstrip to scripts/llmstrip")
        exit(1)
        
    result = subprocess.run([llmstrip_exe, "--mode", "text", args.input], 
                            capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running llmstrip: {result.stderr}")
        humanized = open(args.input, 'r', encoding='utf-8').read()
    else:
        humanized = result.stdout.strip()
    
    save_to_file(humanized, args.output)
    print(f"Humanized text successfully saved to {args.output}")
