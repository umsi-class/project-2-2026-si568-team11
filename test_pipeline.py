import json
import importlib.util

from langchain_pipeline import analyze_tos

spec = importlib.util.spec_from_file_location("pdf_processing_edited", "pdf-processing-edited.py")
pdf_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_processing)

extract_chunks = pdf_processing.extract_chunks

def main():
    pdf_path = "sample-tos/google_tos_example.pdf"
    
    print(f"1. Reading PDF ({pdf_path}) and extracting chunks...")
    with open(pdf_path, "rb") as pdf_file:
        chunks = extract_chunks(input_url=None, pdf=pdf_file)
    
    if not chunks:
        print("Error: Could not extract chunks. Make sure the site allows scraping.")
        return
        
    print(f"Successfully extracted {len(chunks)} chunks!")
    
    print("\n2. Processing chunks through the LangChain pipeline (gpt-4o-mini)...")
    result = analyze_tos(chunks)
    
    print("Finished Analysis:")
    
    # Pretty print the json output
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
