import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

# Define Output Schemas for JsonOutputParser
class SummaryOutput(BaseModel):
    service_introduction: str = Field(description="What is this service? Main functions.")
    user_rights: str = Field(description="What rights do users have? What are the user's obligations? Prohibited behaviors.")
    data_and_privacy: str = Field(description="What data is collected? How is it used? Is it shared?")
    payment_and_refund: str = Field(description="Fee standards, payment methods, refund policy.")
    limitation_of_liability: str = Field(description="In what situations is the service provider not liable?")
    dispute_resolution: str = Field(description="How are disputes resolved (arbitration, litigation, etc.)?")
    other_important_terms: str = Field(description="Such as automatic renewal, service changes, termination conditions.")

class RiskHighlightOutput(BaseModel):
    risks: List[str] = Field(description="List of specific, actionable risks or unfair clauses found in the Terms of Service. Be specific, highlight hidden fees, aggressive arbitration, or invasive data collection.")

def analyze_tos(chunks: List[str]) -> dict:
    """
    Main LangChain agent pipeline function.
    Takes a list of string chunks containing Terms of Service text.
    Returns a unified JSON dictionary containing the summary and risk highlights.
    """
    # 1. Initialize the LLM
    # We are using the GitHub Models endpoint which uses the openai SDK
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        return {"error": "GITHUB_TOKEN not found in environment variables. Please add it to your .env file."}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=github_token,
        base_url="https://models.inference.ai.azure.com"
    )

    # Convert string chunks to LangChain Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # We will combine chunks by joining them. 
    # For gpt-4o-mini, context window is 128k, so joining standard ToS chunks directly into one text usually fits.
    full_text = "\n\n".join(chunk for chunk in chunks)

    # 2. Setup the Summary Prompt (incorporating Zhuofu's design)
    summary_parser = JsonOutputParser(pydantic_object=SummaryOutput)
    summary_template = """
You are a professional legal document analyst, skilled at transforming complex Terms of Service into clear, concise summaries.
Your goal is to help users understand the key points of the Terms of Service in plain language (avoiding legal jargon).

The content to be summarized is:
{text}

{format_instructions}

The summary must be provided according to the actual text content and cannot be fabricated. If a certain part of the terms is not mentioned in the terms, explicitly state that it was not found or omit it.

Add a disclaimer at the end of the final text generation (you don't need to add it to every field): "This summary is for understanding and reference only and does not constitute legal advice. If you have any questions, please consult a professional lawyer."
"""
    summary_prompt = PromptTemplate(
        template=summary_template,
        input_variables=["text"],
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    # 3. Setup the Risk Highlighting Prompt (Jiayi's specific task)
    risk_parser = JsonOutputParser(pydantic_object=RiskHighlightOutput)
    risk_template = """
You are a consumer protection advocate specializing in analyzing Terms of Service and Privacy Policies.
Your goal is to extract and highlight the most critical risks that a user might unknowingly agree to.

The content to be analyzed is:
{text}

Focus specifically on finding:
1. Unexpected or aggressive data sharing/selling.
2. Hidden fees, strict cancellation policies, or automatic renewals.
3. Severe limitations of liability or mandatory binding arbitration.
4. Clauses where the provider can terminate service or change terms without notice.

{format_instructions}
"""
    risk_prompt = PromptTemplate(
        template=risk_template,
        input_variables=["text"],
        partial_variables={"format_instructions": risk_parser.get_format_instructions()}
    )

    # 4. Create the execution chains
    summary_chain = summary_prompt | llm | summary_parser
    risk_chain = risk_prompt | llm | risk_parser

    # 5. Run the pipelines
    try:
        summary_result = summary_chain.invoke({"text": full_text})
        risk_result = risk_chain.invoke({"text": full_text})
    except Exception as e:
        return {"error": f"Failed to process with LangChain: {str(e)}"}

    # 6. Format output clearly for users (combinined into one JSON/Dict)
    final_output = {
        "summary": summary_result,
        "risk_highlights": risk_result.get("risks", []),
        "disclaimer": "This summary is for understanding and reference only and does not constitute legal advice. If you have any questions, please consult a professional lawyer."
    }

    return final_output
