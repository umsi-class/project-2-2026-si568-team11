import json
import importlib.util
import streamlit as st

from langchain_pipeline import analyze_tos

spec = importlib.util.spec_from_file_location("pdf_processing_edited", "pdf-processing-edited.py")
pdf_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf_processing)

extract_chunks = pdf_processing.extract_chunks

def main():
    pdf_path = "sample-tos/google_tos_example.pdf"
    st.title(
            "What are you agreeing to?",
            anchor=False)


    st.subheader(
        "Understanding Terms of Service (ToS)",
        anchor=False,
        divider="blue"
    )
    multi = '''Every day, we're presented with ToS documents that we have to agree to in order to use a service. \
    \nThe language in these documents are confusing and difficult to understand. \
    \nInstead of blindly hitting 'accept', use this tool to understand what exactly you're agreeing to.
    '''
    st.subheader(multi)
    file_upload = st.file_uploader("Upload your file here: ", type="pdf")
    url_variable = st.query_params.get("url", "")
    pasted_url = st.text_input("Or, paste the URL: ", value=url_variable)
    st.query_params["url"] = pasted_url
    print(f"1. Reading PDF ({file_upload}) and extracting chunks...")
    chunks = extract_chunks(input_url=st.query_params["url"], pdf=file_upload)
    
    if not chunks:
        print("Error: Could not extract chunks. Make sure the site allows scraping.")
        return
        
    print(f"Successfully extracted {len(chunks)} chunks!")
    
    print("\n2. Processing chunks through the LangChain pipeline (gpt-4o-mini)...")
    with st.spinner("Reading your ToS document..."):
        result = analyze_tos(chunks)
        
        print("\n" + "="*50)
        print("FINISHED ANALYSIS:")
        print("="*50)
        
        # Pretty print the json output
        print(json.dumps(result, indent=2))
    try:
        service_introduction = result['summary']['service_introduction']
        st.success("Done!")
        user_rights = result['summary']['user_rights']
        data_privacy = result['summary']['data_and_privacy']
        payment_refund = result['summary']['payment_and_refund']
        liability = result['summary']['limitation_of_liability']
        dispute_resolution = result['summary']['dispute_resolution']
        other_terms = result['summary']['other_important_terms']
        
        st.subheader("ToS Summary:")
        intro = f'''#### Description:\
        \n{service_introduction}
        '''
        rights = f'''#### What rights do you have?\
        \n{user_rights}
        '''
        privacy = f'''#### What about your data and privacy?\
        \n{data_privacy}
        '''
        refund = f'''#### What about payment and refunds?\
        \n{payment_refund}
        '''
        limitations = f'''#### What about liability?\
        \n{liability}
        '''
        resolution = f'''#### What about dispute resolution?\
        \n{dispute_resolution}
        '''
        terms = f'''#### Is there anything else I should know about?\
        \n{other_terms}
        '''
        st.markdown(intro)
        st.divider()
        st.markdown(rights)
        st.divider()
        st.markdown(privacy)
        st.divider()
        st.markdown(refund)
        st.divider()
        st.markdown(limitations)
        st.divider()
        st.markdown(resolution)
        st.divider()
        st.markdown(terms)
        st.divider()
        st.warning(result['disclaimer'])
        st.button("Rerun")
    except:
        st.warning("Attempt failed - review error message below.")
        st.markdown(result)
        st.button("Rerun")
    
if __name__ == "__main__":
    main()
