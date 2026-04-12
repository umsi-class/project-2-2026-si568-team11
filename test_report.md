## First test ##

date: Aprial 1

Action: run the test_pipeline.py
result:
{
  "error": "Failed to process with LangChain: Error code: 413 - {'error': {'code': 'tokens_limit_reached', 'message': 'Request body too large for gpt-4o-mini model. Max size: 8000 tokens.', 'details': 'Request body too large for gpt-4o-mini model. Max size: 8000 tokens.'}}"
}
which means the request content is too large and exceeds the model's token limit.

Action: Try using other models to avoid input limitations.
result: Same as above, indicating that all models on GitHub Models are limited to 8000 tokens.

suggestion: modifying the `langchain_pipeline.py` file to add the following functionality:
    Split the entire chunks into 6500-token chunks, ensuring each chunk's input is less than 8000.
    Summarize each chunk separately, and finally merge all results.
    This way, the output will be complete without exceeding the limit.


## Second test ##

date: Aprial 12

Action: run evvaluation_runner.py and output_quality_evalution.py
result: The code runs normally.
    summary total score: 19/21
    risk highlightscore: 27/27
    Both features scored above 90%, indicating good performance, and no modification to the prompts is required.