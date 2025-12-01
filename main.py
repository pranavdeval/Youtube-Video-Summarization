from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def extract_transcript(video_id):

    try:
        youtube_transcript_api = YouTubeTranscriptApi()
        transcript_list = YouTubeTranscriptApi.fetch(youtube_transcript_api, video_id=video_id, languages=["en"])
        transcript_text = " ".join(token.text for token in transcript_list.snippets)

        return transcript_text
    except Exception as e:
        print(f"Exception: {str(e)}")


def split_transcript_to_chunks(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript_text])
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history):
    """Format chat history as a string for context"""
    if not chat_history:
        return ""
    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)


video_id = input("Please Enter the Youtube Video ID: ")

#================= INDEXING ====================

transcript_text = extract_transcript(video_id)

if not transcript_text:
    print("Failed to extract transcript. Exiting.")
    exit(1)

chunks = split_transcript_to_chunks(transcript_text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embeddings)

#================ RETRIEVING ======================

retriever = vector_store.as_retriever(search_kwargs={'k': 4})

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient just say that you don't know.

    Chat History:
    {chat_history}

    Context: {context}
    
    Question: {question}
    """,
    input_variables=['context', 'question', 'chat_history']
)

model = ChatOpenAI(model="gpt-4o-mini")  # Changed to a more commonly available model

parser = StrOutputParser()

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        break

    # Fix: Pass the question as a string, not the entire chat history
    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda x: x['question']) | retriever | RunnableLambda(format_docs),
        'question': RunnableLambda(lambda x: x['question']),
        'chat_history': RunnableLambda(lambda x: x['chat_history'])
    })

    main_chain = parallel_chain | prompt | model | parser

    result = main_chain.invoke({
        'question': user_input,
        'chat_history': format_chat_history(chat_history)
    })

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result))

    print(f"AI: {result}")
