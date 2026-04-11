import json
import os
from pdf_processing import extract_chunks
from langchain_pipeline import analyze_tos

REQUIRED_TOP_KEYS = ["summary", "risk_highlights", "disclaimer"]

REQUIRED_SUMMARY_KEYS = [
    "service_introduction",
    "user_rights",
    "data_and_privacy",
    "payment_and_refund",
    "limitation_of_liability",
    "dispute_resolution",
    "other_important_terms"
]

TEST_FILES = [
    "sample-tos/google_tos_example.pdf"
]


def check_output_structure(result):
    """
    Check whether the final output has the expected structure.
    Returns a list of issues. If empty, structure is valid.
    """
    issues = []

    for key in REQUIRED_TOP_KEYS:
        if key not in result:
            issues.append(f"Missing top-level key: {key}")

    if "summary" in result:
        if not isinstance(result["summary"], dict):
            issues.append("summary is not a dictionary")
        else:
            for key in REQUIRED_SUMMARY_KEYS:
                if key not in result["summary"]:
                    issues.append(f"Missing summary key: {key}")

    if "risk_highlights" in result and not isinstance(result["risk_highlights"], list):
        issues.append("risk_highlights is not a list")

    if "disclaimer" in result and not isinstance(result["disclaimer"], str):
        issues.append("disclaimer is not a string")

    return issues


def evaluate_pdf(pdf_path):
    """
    Run extraction + analysis for one PDF file.
    """
    print(f"Evaluating file: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"FAILED: File not found -> {pdf_path}")
        return None

    try:
        with open(pdf_path, "rb") as pdf_file:
            chunks = extract_chunks(input_url=None, pdf=pdf_file)
    except Exception as e:
        print(f"FAILED during PDF reading/extraction: {e}")
        return None

    if not chunks:
        print("FAILED: No text chunks extracted.")
        return None

    print(f"Chunk extraction passed: {len(chunks)} chunks extracted")

    try:
        result = analyze_tos(chunks)
    except Exception as e:
        print(f"FAILED during LangChain analysis: {e}")
        return None

    if not isinstance(result, dict):
        print("FAILED: Output is not a dictionary")
        return None

    if "error" in result:
        print("FAILED: Pipeline returned an error")
        print(result["error"])
        return None

    issues = check_output_structure(result)

    if issues:
        print("STRUCTURE CHECK FAILED")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("STRUCTURE CHECK PASSED")

    print(f"Risk highlights count: {len(result.get('risk_highlights', []))}")

    output_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_evaluation_output.json"

    with open(output_name, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved output to: {output_name}")

    return {
        "file": pdf_path,
        "num_chunks": len(chunks),
        "num_risks": len(result.get("risk_highlights", [])),
        "structure_passed": len(issues) == 0,
        "issues": issues
    }


def main():
    all_results = []

    for pdf_path in TEST_FILES:
        summary = evaluate_pdf(pdf_path)
        if summary is not None:
            all_results.append(summary)

    print("FINAL EVALUATION SUMMARY")


    if not all_results:
        print("No successful evaluation runs.")
        return

    for item in all_results:
        print(f"File: {item['file']}")
        print(f"Chunks extracted: {item['num_chunks']}")
        print(f"Risk highlights: {item['num_risks']}")
        print(f"Structure passed: {item['structure_passed']}")
        if item["issues"]:
            print("Issues:")
            for issue in item["issues"]:
                print(f"- {issue}")
        print()

    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("Saved summary to: evaluation_summary.json")


if __name__ == "__main__":
    main()