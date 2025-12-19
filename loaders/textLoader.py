from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def textLoader(path:str):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return docs

def listPdfs(path_directory):
    dir_pdf = Path(path_directory)

    pdf_files = list(dir_pdf.glob("**/*.pdf"))
    return pdf_files

def directoryLoader(path:str):
    loader = DirectoryLoader(
        path=path,
        loader_cls= PyPDFLoader,
        glob="**/*.pdf",
        show_progress=False,

    )
  
    return loader.load()


def splitDocs(docs, chunk_size=1000, chunk_overlap=400):
    textSplit = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = textSplit.split_documents(docs)
    return chunks



