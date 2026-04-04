import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field, SecretStr
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class SummaryOutput(BaseModel):
    service_introduction: str = Field(description="What is this service? Main functions.")
    user_rights: str = Field(description="What rights do users have? What are the user's obligations? Prohibited behaviors.")
    data_and_privacy: str = Field(description="What data is collected? How is it used? Is it shared?")
    payment_and_refund: str = Field(description="Fee standards, payment methods, refund policy.")
    limitation_of_liability: str = Field(description="In what situations is the service provider not liable?")
    dispute_resolution: str = Field(description="How are disputes resolved (arbitration, litigation, etc.)?")
    other_important_terms: str = Field(description="Such as automatic renewal, service changes, termination conditions.")


class RiskHighlightOutput(BaseModel):
    risks: List[str] = Field(
        description="List of specific, actionable risks or unfair clauses found in the Terms of Service."
    )


# Count tokens for a given text string so we can keep each request under the model limit.
def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# Batch the extracted chunks into smaller groups based on a token threshold.
# This prevents the full request body from exceeding the GitHub Models 8k limit.
# We use 5500 tokens as a safer upper bound because prompt instructions and JSON formatting also use tokens.
def batch_chunks_by_token_limit(
    chunks: List[str],
    max_tokens: int = 5500,
    model_name: str = "gpt-4o-mini"
) -> List[str]:
    batched = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, model_name)

        # If a single extracted chunk is already too large, split it further by paragraph.
        # This adds robustness for unusually large source chunks.
        if chunk_tokens > max_tokens:
            paragraphs = chunk.split("\n")
            temp_batch = []
            temp_tokens = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_tokens = count_tokens(para, model_name)

                if temp_batch and temp_tokens + para_tokens > max_tokens:
                    batched.append("\n\n".join(temp_batch))
                    temp_batch = [para]
                    temp_tokens = para_tokens
                else:
                    temp_batch.append(para)
                    temp_tokens += para_tokens

            if temp_batch:
                batched.append("\n\n".join(temp_batch))
            continue

        if current_batch and current_tokens + chunk_tokens > max_tokens:
            batched.append("\n\n".join(current_batch))
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens

    if current_batch:
        batched.append("\n\n".join(current_batch))

    return batched

# Merge multiple batch-level summary outputs into one final summary dictionary.
# Since each batch is summarized independently, we need a post-processing step to combine them.
def merge_summary_results(summary_results: List[dict]) -> dict:
    merged = {
        "service_introduction": [],
        "user_rights": [],
        "data_and_privacy": [],
        "payment_and_refund": [],
        "limitation_of_liability": [],
        "dispute_resolution": [],
        "other_important_terms": []
    }

    for result in summary_results:
        for key in merged:
            value = result.get(key, "").strip()

            # Avoid adding duplicate text across batches.
            if value and value not in merged[key]:
                merged[key].append(value)

    final_summary = {}
    for key, values in merged.items():
        if values:
            final_summary[key] = "\n\n".join(values)
        else:
            final_summary[key] = "Not found in the provided text."

    return final_summary


# Remove duplicate risks collected from different batches while preserving their original order.
def deduplicate_risks(risks: List[str]) -> List[str]:
    deduped = []
    seen = set()

    for risk in risks:
        risk_clean = risk.strip()
        if risk_clean and risk_clean not in seen:
            deduped.append(risk_clean)
            seen.add(risk_clean)

    return deduped


def analyze_tos(chunks: List[str]) -> dict:
    """
    Main LangChain pipeline function.

    NEW FUNCTIONALITY ADDED:
    - Instead of sending the entire ToS in one request, this version processes the text in token-limited batches.
    - This solves the previous 413 'request body too large' error from GitHub Models.
    - The function now merges batch-level summaries and deduplicates risk outputs before returning the final JSON.
    """
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        return {
            "error": "GITHUB_TOKEN not found in environment variables. Please add it to your .env file."
        }

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=SecretStr(github_token),
        base_url="https://models.inference.ai.azure.com"
    )


    # Split the input into safe-size batches before sending requests to the model.
    text_batches = batch_chunks_by_token_limit(
        chunks=chunks,
        max_tokens=5500,
        model_name="gpt-4o-mini"
    )

    summary_parser = JsonOutputParser(pydantic_object=SummaryOutput)
    summary_template = """
You are a professional legal document analyst, skilled at transforming complex Terms of Service into clear, concise summaries.
Your goal is to help users understand the key points of the Terms of Service in plain language (avoiding legal jargon).

The content to be summarized is:
{text}

{format_instructions}

The summary must be provided according to the actual text content and cannot be fabricated.
If a certain part of the terms is not mentioned in this section, explicitly state "Not found in this section."
"""
    summary_prompt = PromptTemplate(
        template=summary_template,
        input_variables=["text"],
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

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

    summary_chain = summary_prompt | llm | summary_parser
    risk_chain = risk_prompt | llm | risk_parser

    partial_summaries = []
    all_risks = []

    # Run the summary and risk extraction pipeline batch by batch.
    # This is the main change that makes long documents processable.
    for i, batch_text in enumerate(text_batches):
        try:
            print(
                f"Processing batch {i + 1}/{len(text_batches)} "
                f"({count_tokens(batch_text, 'gpt-4o-mini')} tokens)"
            )

            summary_result = summary_chain.invoke({"text": batch_text})
            risk_result = risk_chain.invoke({"text": batch_text})

            partial_summaries.append(summary_result)
            all_risks.extend(risk_result.get("risks", []))

        except Exception as e:
            return {"error": f"Failed on batch {i + 1}: {str(e)}"}

    # Merge all batch-level outputs into one final structured response.
    merged_summary = merge_summary_results(partial_summaries)
    deduped_risks = deduplicate_risks(all_risks)

    final_output = {
        "summary": merged_summary,
        "risk_highlights": deduped_risks,
        "disclaimer": "This summary is for understanding and reference only and does not constitute legal advice. If you have any questions, please consult a professional lawyer."
    }

    return final_output