import argparse
import json
import urllib.request
import os

def humanize_text(text):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "phi3",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert text humanizer. Your only job is to rewrite the user's text to sound natural, "
                    "human-written, and engaging. Remove common AI buzzwords (like 'novel', 'robust', 'delve', 'leverage'), "
                    "vary the sentence structure, and maintain all technical accuracy. "
                    "Output ONLY the rewritten text. Do not include introductory phrases, conversational filler, or repeat the text."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("message", {}).get("content", "").strip()
    except urllib.error.URLError:
        print("Warning: Could not connect to Ollama on http://localhost:11434.")
        print("Please ensure Ollama is running and the phi3 model is pulled.")
        print("Falling back to original text.")
        return text
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return text

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
    parser = argparse.ArgumentParser(description="Humanize AI generated text via local Ollama.")
    parser.add_argument('--input', type=str, required=True, help='Path to a file containing raw text')
    parser.add_argument('--output', type=str, required=True, help='Path to save output file (.md or .docx)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        exit(1)
        
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    print("Humanizing text with local phi3 model...")
    humanized = humanize_text(raw_text)
    
    save_to_file(humanized, args.output)
    print(f"Humanized text successfully saved to {args.output}")
