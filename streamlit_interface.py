import streamlit as st

st.title(
    "What are you agreeing to?",
    anchor=False)
st.subheader(
    "Understanding Terms of Service (ToS)",
    anchor=False,
    divider="blue"
)
multi = '''Every day, we're presented with ToS documents that we have to agree to in order to use a service.

The language in these documents are confusing and difficult to understand. Instead of blindly hitting 'accept', use this tool to understand what exactly you're agreeing to.
'''
st.markdown(multi)
file_upload = st.file_uploader("Upload your file here: ", type="pdf")
file_paste = st.text_input("Or, paste the contents of the document here: ")
