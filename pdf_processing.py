import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_chunks(input_url, pdf=None):
    """
    TODO: write docstring
    """

    url_text = ""
    # Grab url text if user gives url
    if input_url:

        # catching the case where a user only enters youtube.com instead of https://youtube.com
        if not input_url.lower().startswith(("http://", "https://")):
            input_url = "https://" + input_url

        try:
            headers = {
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.102 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/'
            }
            session = requests.Session()
            response = session.get(input_url, headers=headers)
            response.raise_for_status()  #added this for response check
            soup_val = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup_val.find_all('p')
            url_text = '\n'.join(p.get_text() for p in paragraphs if p.get_text().strip())
        except:
            # TODO: display a message on interface to say url failed
            pass

    # Extract text from PDF or URL
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        current_text = ""
        for currPage in pdf_reader.pages:
            current_text += currPage.extract_text()
        store_name = pdf.name.split('.')[0]
    elif url_text:
        current_text = url_text
        store_name = input_url.replace("https://", "").replace("http://", "").replace("/", "_")
    else:
        # TODO: display a message on interface to ask user to upload valid PDF/URL
        return

    # Split text into chunks specifically for langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text=current_text)

    return chunks
