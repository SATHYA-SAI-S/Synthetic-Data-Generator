# Antigravity Rules for Synthetic-Data-Generator

## Rule: AI Document Humanizer Pipeline
**Trigger**: `always_on`
**Description**: Forces the agent to use the local humanizer pipeline before saving any explanation documentation.

Whenever the user asks you to write, generate, or draft an explanation, documentation, or report into a `.md` or `.docx` file, you MUST NOT write the final file directly. 

Instead, you MUST strictly follow this pipeline:
1. Write your raw explanation draft to a temporary markdown file in the `scratch/` directory (e.g., `scratch/draft_explanation.md`).
2. Run the humanizer script via the command line: 
   `.venv310\Scripts\python.exe scripts\humanize_doc.py --input scratch\draft_explanation.md --output <final_requested_file_path>`
3. Wait for the command to finish. The script will automatically communicate with the local Ollama `phi3` model to humanize the text and save it to the final destination (supporting both `.md` and `.docx`).
4. (Optional) Clean up the temporary draft file.

Never bypass this pipeline for user-facing documentation unless explicitly asked to skip humanization.
