import sys
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf'])
    import pypdf

reader = pypdf.PdfReader("MidtermProject.pdf")
text = ""
for i, page in enumerate(reader.pages):
    text += f"\n--- Page {i+1} ---\n"
    text += page.extract_text() + "\n"
with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("done")
