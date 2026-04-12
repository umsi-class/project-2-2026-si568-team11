# Project 2: GenAI Group Project

**Implementation Idea**: An interface to help users understand terms of service (that they can upload into our tool), and use an LLM to give a summary, highlight potential risks, and provide any other helpful information that we see fit.

**Pipeline Implementation Steps**: 
(Refer to 'langchain_pipeline.py', 'pdf-processing-edited.py', 'output_quality_evaluation.py', 'test_pipeline.py')
- Step-1: Build a pipeline to analyze Terms of Service (ToS) using a language model and return structured summaries and risk highlights.
- Step-2: Define Pydantic schemas to standardize outputs like service overview, user rights, privacy, payments, and risks.
- Step-3: Large input text is split into token-limited batches to avoid API limits, using a custom token counting and batching function.
- Step-4: Each batch is processed through LangChain prompts to generate summaries and extract potential risks, then results are merged and deduplicated.
- Step-5: The final output is a clean JSON containing the combined summary, unique risks, and a disclaimer, with error handling for missing API keys or batch failures.
  
**Web Interface using Streamlit**: 
(Refer to 'streamlit_app.py')
- Step-1: Build a Streamlit web app that lets users upload a PDF or URL of Terms of Service and extracts text chunks from it.
- Step-2: Send those chunks to the analyze_tos pipeline (LLM-based) to generate structured summaries and risk insights.
- Step-3: The results are displayed in a clean UI with sections (rights, privacy, payments, etc.), along with error handling and a rerun option.

**GenAI Use & Prompt Summary**:
- [Reference](https://docs.google.com/document/d/1yY6FB5KD646qOpsB_r3-uLaQhtLoYEksJ7qhC5-PhJg/edit?tab=t.0)

**Workflow with Code modules / files**:

- Input Layer (User / Test Runner):
  Modules:
    1. streamlit app file (UI upload + URL input), adapted from test_pipeline.py 
    2. evaluation_runner (CLI testing)
       
- Data Extraction (extract_chunks) -> (PDF reading + web scraping + chunking)
  Module:
    1. pdf_processing.py (and edited version)
       
- Preprocessing (Token Management)
  Module:
    1. langchain_pipeline.py
  Functions:
    1. count_tokens()
    2. batch_chunks_by_token_limit()
       
- LLM Processing (analyze_tos)
  Module:
    1. langchain_pipeline.py
Components:
    1. ChatOpenAI (LLM call)
    2. Prompt templates (summary + risk)
    3. LangChain chains
       
- Post-processing
  Module:
    1. langchain_pipeline.py
  Functions:
    1. merge_summary_results()
    2. deduplicate_risks()
       
- Output Generation
  Module:
    1. langchain_pipeline.py → returns final JSON
    2. streamlit app file → displays results
    3. test/evaluation scripts → save JSON output
       
- Evaluation & Validation Layer
  Module:
    1. evaluation script (output_quality_checks.py)
    2. evaluation_runner / test file
  Functions:
    1. check_output_structure()
    2. check_summary_accuracy()
    3. check_risk_highlight_usefulness()
