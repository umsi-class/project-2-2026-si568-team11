
# pip install -U langchain
# pip install -U langchain-openai
# pip install -U langchain-anthropic
# pip install langchain langchain-community langchain-openai langchain-text-splitters

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. laod file
loader = TextLoader("example.txt")
documents = loader.load()

# if use pdf, you can use PyPDFLoader
# pip install pypdf
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("example.pdf")
# documents = loader.load()

# 2. split text
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. prepare prompt
template_str = """
You are a professional legal document analyst, skilled at transforming complex Terms of Service into clear, concise summaries.
Your goal is to help users understand the key points of the Terms of Service in plain language (avoiding legal jargon).

The content to be summarized is:
{text}

Please output the summary in the following format (Output only JSON, with no other content.):
- Service Introduction: What is this service? Main functions.
- User Rights: What rights do users have? What are the user's obligations? Prohibited behaviors.
- Data and Privacy: What data is collected? How is it used? Is it shared?
- Payment and Refund: Fee standards, payment methods, refund policy.
- Limitation of Liability: In what situations is the service provider not liable?
- Dispute Resolution: How are disputes resolved (arbitration, litigation, etc.)?
- Other Important Terms: Such as automatic renewal, service changes, termination conditions.

The summary must be provided according to the actual text content and cannot be fabricated. If a certain part of the terms is not mentioned in the terms, omit that part.

Add a disclaimer at the end of the summary: "This summary is for understanding and reference only and does not constitute legal advice. If you have any questions, please consult a professional lawyer."
"""

prompt = PromptTemplate.from_template(template_str)

# 4. model
# don't forget to set your OPENAI_API_KEY in the environment variables

# pip install python-dotenv

#from dotenv import load_dotenv
# import os
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 5. load chain（LCEL）
chain = prompt | llm | (lambda x: x.content)

# 6. run
full_text = "\n".join([chunk.page_content for chunk in chunks])
result = chain.invoke({"text": full_text[:4000]})
print(result)