# Project 2: GenAI Group Project

**Implementation Idea**: An interface to help users understand terms of service (that they can upload into our tool), and use an LLM to give a summary, highlight potential risks, and provide any other helpful information that we see fit.

**Pipeline Implementation Steps**: (refer to 'langchain_pipeline.py', 'pdf-processing-edited.py', 'output_quality_evaluation.py', 'test_pipeline.py')
- Step-1: Build a pipeline to analyze Terms of Service (ToS) using a language model and return structured summaries and risk highlights.
- Step-2: Define Pydantic schemas to standardize outputs like service overview, user rights, privacy, payments, and risks.
- Step-3: Large input text is split into token-limited batches to avoid API limits, using a custom token counting and batching function.
- Step-4: Each batch is processed through LangChain prompts to generate summaries and extract potential risks, then results are merged and deduplicated.
- Step-5: The final output is a clean JSON containing the combined summary, unique risks, and a disclaimer, with error handling for missing API keys or batch failures.
  
**Web Interface using Streamlit**: ('refer to 'streamlit_app.py' & streamlit_interface.py)
- Step-1: Build a Streamlit web app that lets users upload a PDF or URL of Terms of Service and extracts text chunks from it.
- Step-2: Send those chunks to the analyze_tos pipeline (LLM-based) to generate structured summaries and risk insights.
- Step-3: The results are displayed in a clean UI with sections (rights, privacy, payments, etc.), along with error handling and a rerun option.

**GenAI Use & Prompt Summary**:
- Reference: https://docs.google.com/document/d/1yY6FB5KD646qOpsB_r3-uLaQhtLoYEksJ7qhC5-PhJg/edit?tab=t.0
