from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import qdrant_client
import sys
import logging
import os
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import streamlit as st

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

PAGE_TITLE: str = "FCA Handbook Q&A"
PAGE_ICON: str = "🤖"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

if 'key' not in st.session_state:
    st.write("""
    To start working with the application you need to use your OpenAI API key. You can get one at https://platform.openai.com/account/api-keys.
    Note: the application stores the key only for the duration of the session. However, it's better to delete the key when it is no longer needed.
    """)
    key = st.text_input(label='OPENAI_API_KEY:', type='password')
    if len(key) > 0 and key[:2] == 'sk':
        st.session_state.key = key
        st.experimental_rerun()
    st.stop()


def init_session() -> None:
    st.session_state.context = None
    st.session_state.question = None
    st.session_state.sub_questions = None
    st.session_state.end = False

    st.session_state.messages = []


if 'question' not in st.session_state:
    init_session()


def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    # print([message for message in st.session_state.messages])


def show_chat() -> None:
    if st.session_state.messages:
        for i in range(len(st.session_state.messages)):
            msg = st.session_state.messages[i]
            content = st.session_state.messages[i].content
            if type(msg) is HumanMessage:
                content = "**" + content + "**"
            st.write(content)


def show_question_input():
    def submit_context():
        if len(st.session_state.user_context_widget) != 0:
            st.session_state.context = st.session_state.user_context_widget
        else:
            logging.info("empty user context")

    def submit_question():
        if len(st.session_state.user_question_widget) != 0:
            st.session_state.question = st.session_state.user_question_widget
        else:
            logging.info("empty user question")

    st.text_area(label="Context",
                 help="describe the important details of the situation you are investigating",
                 key='user_context_widget',
                 value="" if st.session_state.context is None else st.session_state.context,
                 on_change=submit_context)
    st.text_input(label='User Question:',
                  key='user_question_widget',
                  value="" if st.session_state.question is None else st.session_state.question,
                  on_change=submit_question)
    st.button("Submit")


@st.cache_resource
def get_retriever(api_key):
    client = qdrant_client.QdrantClient(
        path="./qdrant.vdb", prefer_grpc=True
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
    qdrant = Qdrant(
        client=client, collection_name="handbook",
        embeddings=embeddings
    )
    retriever = qdrant.as_retriever(search_type="mmr")  # k=2, fetch_k=10
    return retriever


def get_llm(model_name, model_temperature, api_key, max_tokens=256):
    if model_name == "text-davinci-003":
        return OpenAI(temperature=model_temperature, model_name=model_name, max_tokens=max_tokens,
                      openai_api_key=api_key)
    else:
        return ChatOpenAI(temperature=model_temperature, model_name=model_name, max_tokens=max_tokens,
                          openai_api_key=api_key)


def qa(context, question, model_name, api_key, retriever):
    llm = get_llm(model_name=model_name, model_temperature=0, api_key=api_key, max_tokens=256)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    query = \
        f"""
    You are an experienced FCA regulator and your goal is to asses if the company is compliant with regulations. If there is a chance that company is in breach of regulations, you have to provide advice for improvement.
    Explain your reasoning step by step to achieve clear and precise explanation.

    Situation: ``` {context} ```
    Question: ``` {question} ```
    """
    response = qa.run(query)
    return response


def get_sub_questions(context, question, model_name, api_key):
    llm = get_llm(model_name=model_name, model_temperature=0, api_key=api_key, max_tokens=1024)
    format_instructions = \
        """
        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing \\"```json\\" and \\"```\\":
        ```json{
        "q": [
        { 
            "chapter": string // What is the chapter of the FCA Handbook that most probably have answer to the question?
            "question": string // What is the sub-question to clarify?
        }, ... 
        ]}
        ``` 
        """

    template = \
        """
        You are an experienced FCA regulator and your goal is to asses if the company is compliant with regulations. 
        For the following situation and question, extract key sub-questions related to specific chapters of the FCA Handbook and break down the original questions into a series of follow up questions using the format: 
        {format_instructions} 
    
        Explain your reasoning step by step.  
    
        Situation: ``` {context} ```
        Question: ``` {question} ```
        """

    prompt = ChatPromptTemplate.from_template(template=template)
    messages = prompt.format_messages(context=context, question=question, format_instructions=format_instructions)
    logging.info(messages)
    response = llm(messages)
    logging.info(response)

    output_parser = StructuredOutputParser.from_response_schemas([])
    output_dict = output_parser.parse(response.content)
    return output_dict['q']


def qa_sub_question(context, question, model_name, api_key, retriever):
    logging.info(question)
    return [HumanMessage(content=question), AIMessage(content=qa(context, question, model_name, api_key, retriever))]


def get_final_answer(context, question, messages, model_name, api_key):
    prompts = []
    prompts.append(SystemMessage(
        content='You are an experienced FCA regulator and your goal is to asses if the company is compliant with regulations. You are able to deal with ambuigity in regulations.'))
    prompts.extend(messages)

    prompts.append(HumanMessage(
        content= \
            f"""
        Use the dialog above and the following pieces of context to answer the question at the end. If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer.
        Explain your reasoning step by step to achieve clear and precise explanation. Format output text using Markdown. 
    
        Situation: ``` {context} ```
        Question: ``` {question} ```
        """))

    llm = get_llm(model_name=model_name, model_temperature=0, api_key=api_key, max_tokens=None)
    response = llm(prompts)
    logging.info(response)
    return response.content


def main() -> None:
    if st.session_state.context is None or st.session_state.question is None:
        show_question_input()
    elif not st.session_state.end:
        context = st.session_state.context
        question = st.session_state.question

        st.write(context)
        st.write(question)
        st.divider()
        show_chat()

        if st.session_state.sub_questions is None:
            with st.spinner('Breaking down the question...'):
                st.session_state.sub_questions = get_sub_questions(context, question, 'gpt-3.5-turbo', st.session_state.key)
                st.experimental_rerun()
        elif len(st.session_state.sub_questions) > 0:
            question = st.session_state.sub_questions.pop(0)
            question = question['question']
            with st.spinner(f'Answering the question: {question}'):
                retriever = get_retriever(st.session_state.key)
                response = qa_sub_question(context, question, 'text-davinci-003', st.session_state.key, retriever)
                logging.info(response)
                st.session_state.messages.extend(response)
                st.experimental_rerun()
        else:
            messages = st.session_state.messages
            response = get_final_answer(context, question, messages, 'gpt-3.5-turbo-16k', st.session_state.key)
            st.session_state.messages.append(AIMessage(content=response))

            st.subheader("Answer:")
            st.divider()
            st.write(response)

            st.session_state.end = True


if __name__ == "__main__":
    main()
