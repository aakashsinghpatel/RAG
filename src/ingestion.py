import config

def load_documents():
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
    try:
        directory_loader = DirectoryLoader(config.PDF_PATH, 
                                    glob="*.pdf", 
                                    loader_cls=PyMuPDFLoader,
                                    silent_errors=False)
        pdf_documents = directory_loader.load()
        return pdf_documents
    except  Exception as e:
        print(f"Error loading documents: {e}")
        return []

def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)