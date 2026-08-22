import os, time, subprocess
while not os.path.exists('docs/Final_Presentation_Master.docx'):
    time.sleep(5)
print("All files exist. Running formatter...")
subprocess.run([".venv310\\\\Scripts\\\\python.exe", "scratch/format_docx.py"])
print("Formatting complete.")
