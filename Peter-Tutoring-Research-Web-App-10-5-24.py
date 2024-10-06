import streamlit as st
    # Streamlit = Open source framework for machine learning / data science.
    # Enables creation of web applications and other machine learning or data science projects. 
    # Imports = Make all functions, variables and classes in module available. 
    # as = Alias/shorthand for module. Purpose-Convenience: Shorten long module name in this case to just st. 
    #                                  Avoiding Conflicts: Simple unique module name that is not the same as another 
    # module. 

from dotenv import load_dotenv
# Allows you to import function from dotenv library (specifically .env file) for managing senstive information securely. (These include Application Programming Interface Keys)

from langchain.llms import OpenAI
    # Allows you to import the OpenAI class from the Langchain library. (Allows you to import and utilize openAI 
    # language models.) 
    #langchain = library / framework. (Files within the langchain are called modules and modules are equivalent 
    # to a single file.)
from langchain.utilities import WikipediaAPIWrapper 
    # Imports a Langchain utilities. Purpose is to extract data from wikipedia specifically and use this to 
    # gain insights.  

from langchain.prompts import PromptTemplate
    # Used for making prompts for the language models. 
    #                                  Purpose of prompts: 1. Summarization, 
    #                                                      2. question answering, 
    #                                                      3. content generation, 
    #                                                      4. context(previous datasets), 
    #                                                      5. structure (structured format for inputs and outputs.)
    #                                                      6. customization and control (Different AI model 
    #                                                               purpose = different prompt logic)

from langchain.memory import ConversationBufferMemory
#Maintains conversation context in interaction with ai model from langchain.chains import LLMChain.

from langchain.agents import Tool
# Imports tool class which can be utilized by a language model to perform a task. 
# Submodule langchain.agents is the file path to do the above process. 

from langchain.agents import AgentType
# Calls on different types of agents in the Langchain framework. This is contingent on the type of task 
# we ask the model to perform. 

from langchain.embeddings.openai import OpenAIEmbeddings
# Enables you to make embeddings using the OpenAI models. 
# Embeddings = Vector space of text where similar meanings are clustered together to provide context for future data. 

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# Imports a tranformer to manipulate embeddings. Manipulates them to relate to sentences. 

from langchain import OpenAI
# Similar to code line 11 but is more generalized. It calls on the entire module that OpenAI provides 
# as opposed to the specific .llms submodule 

from langchain.agents import initialize_agent
# Allows you to initialize an agent to be used in the langchain framework. (initialize is a function 
# that allows the agent to use "tools" or classes within different modules to perform different tasks.)

from langchain.tools import DuckDuckGoSearchRun
# Allows agent to use DuckDuckGoSearchRun class to scrape data from different websites depending on 
# the research topic at hand. 

from langchain.utilities import WikipediaAPIWrapper
# Scrapes data from wikipedia for relevent data.

from langchain.tools import YouTubeSearchTool
# Scrapes data from Youtube and provides useful URLs for the queries you ask. 

from langchain.vectorstores import Chroma
# Vector store. Manage vectors of text data. (Used for similarity search and improving LLM.)

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# Allows the agent to answer questions based on the context of the previous data presented to it. Similar to 
# "from langchain.memory import ConversationBufferMemory" , "from langchain.embeddings.openai import OpenAIEmbeddings"
# and "from langchain.prompts import PromptTemplate" in providing context for the agent. 
# This module specializes in questions and answers. 

from youtubesearchpython import VideosSearch
# Enables agents to scrape youtube metadata and URLS for relevent data. 

from langchain.chains import VectorDBQA
# Similar to "from langchain.chains import RetrievalQA, ConversationalRetrievalChain" for Q and A. 
# Utilizes vector database for pertinent data. 

from langchain.retrievers import SelfQueryRetriever
# Allows the agent to find relevent info based on questions it asks itself. 

from langchain.text_splitter import CharacterTextSplitter
# Optimizes text processing by segementing characters. 

from langchain.chains.question_answering import load_qa_chain
# Allows creation of systems to answers questions based on data. 

from langchain.tools import PubmedQueryRun
# Allows agent to used data from Pubmed to apply to answering questions given to the agent. 

from langchain import LLMMathChain
# Facilitates mathematical computations for answers. 

import sqlite3
# Can now access, make and change SQLite databases. 

import pandas as pd
# Pandas = Data structure library for analysis and data manipulation. 

import os
# Allows the software to connect to certain directories/file paths on certain os's for processing. 

# TODO: Allow users to upload their own files

def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
    # Connects to the SQLite database and creates a new table named 'Research' if it doesn't exist. 
    # Each row is a new column in the table that is generated. 
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT,
                prev_ai_research TEXT
            )
        """)
    # Creates research database function "create_research_db and uses the following SQL 
    # lines to not only connext to the database, but creates a table if does not exist 
    # with the following inputs if they are provided."
def create_messages_db():
    pass
# Creates new function titled "create_messages_db".  Does not have parameters. 
# Does not have any active operation and is here as a placeholder for future use. 
# pass is used syntatically so the code can be read. 
# There are no operations or library references for this function. 
# Mean averages of the message data base can be refined for future use 
# so the AI responses are more accurate and meaningful.

def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df
# Uses SQL query to scan research data table. Scrapes the data from the columns and organizes
# it so it can be processed for future computation. 

def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research):
    # Function insert_research is defined with the following parameters.
    with sqlite3.connect('MASTER.db') as conn:
    # Connects to the SQLite database and inserts new data into the research table.
        cursor = conn.cursor()
    # Creates a new cursor object and executes SQL commands.
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research))    

def generate_research(userInput):
    global tools
    llm=OpenAI(temperature=0.7)
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    pubmed = PubmedQueryRun()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    # Initializes the necessary tools and LLM chain.

    tools = [
        Tool(
            name = "Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for researching information on wikipedia"
        ),
        Tool(
            name = "Duck Duck Go Search Results Tool",
            func = DDGsearch.run,
            description="Useful for search for information on the internet"
        ),
        Tool(
            name = "YouTube Search Tool",
            func = YTsearch.run,
            description="Useful for gathering links on YouTube"
        ),
        Tool(
            name ='Calculator and Math Tool',
            func=llm_math_chain.run,
            description='Useful for mathematical questions and operations'
        ),
        Tool(
            name='Pubmed Science and Medical Journal Research Tool',
            func=pubmed.run,
            description='Useful for Pubmed science and medical research\nPubMed comprises more than 35 million 
            citations for biomedical literature from MEDLINE, life science journals, and online books. Citations 
            may include links to full text content from PubMed Central and publisher web sites.'

        )

    ]
    if st.session_state.embeddings_db:
        qa = VectorDBQA.from_chain_type(llm=llm,
                                        vectorstore=st.session_state.embeddings_db)
        tools.append(
            Tool(
                name='Vector-Based Previous Resarch Database Tool',
                func=qa.run,
                description='Provides access to previous research results'
            )
        )
# Takes information from the previous sources and creates tools to make LLM chain. Vectors 
# Stores chat history with user while also having data from research table available. 

    memory = ConversationBufferMemory(memory_key="chat_history")
    runAgent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                verbose=True, 
                                memory=memory,
                                )
# The previous tools are initialized here and will be modified in streamlit in real time to see the output. 

    with st.expander("Generative Results", expanded=True):
        st.subheader("User Input:")
        st.write(userInput)

        st.subheader("Introduction:")
        with st.spinner("Generating Introduction"):
            intro = runAgent(f'Write an academic introduction about {userInput}')
            st.write(intro['output'])

        st.subheader("Quantitative Facts:")
        with st.spinner("Generating Statistical Facts"):

            quantFacts = runAgent(f'''
                Considering user input: {userInput} and the intro paragraph: {intro} 
                \nGenerate a list of 3 to 5 quantitative facts about: {userInput}
                \nOnly return the list of quantitative facts
            ''')
            st.write(quantFacts['output'])

        prev_ai_research = ""
        if st.session_state.embeddings_db:
            st.subheader("Previous Related AI Research:")
            with st.spinner("Researching Pevious Research"):
                qa = VectorDBQA.from_chain_type(llm=llm,
                                                vectorstore=st.session_state.embeddings_db)
                prev_ai_research = qa.run(f'''
                    \nReferring to previous results and information, write about: {userInput}
                ''')
                st.write(prev_ai_research)

        st.subheader("Recent Publications:")
        with st.spinner("Generating Recent Publications"):
            papers = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 2 to 3 recent academic papers relating to {userInput}.
                \nInclude Titles, Links, Abstracts. 
            ''')
            st.write(papers['output'])

        st.subheader("Reccomended Books:")
        with st.spinner("Generating Reccomended Books"):
            readings = runAgent(f'''
                Consider user input: "{userInput}".
                \nConsider the intro paragraph: "{intro}",
                \nConsider these quantitative facts "{quantFacts}"
                \nNow Generate a list of 5 relevant books to read relating to {userInput}.
            ''')
            st.write(readings['output'])

        st.subheader("YouTube Links:")
        with st.spinner("Generating YouTube Links"):
            search = VideosSearch(userInput)
            ytlinks = ""
            for i in range(1,6):
                ytlinks += (str(i) + ". Title: " + search.result()['result'][0]['title'] + "Link: 
                            https://www.youtube.com/watch?v=" + search.result()['result'][0]['id']+"\n")
                search.next()
            st.write(ytlinks)

        # TODO: Influential Figures

        # TODO: AI Scientists Perscpective

        # TODO: AI Philosophers Perspective

        # TODO: Possible Routes for Original Research
        
        insert_research(userInput, intro['output'], quantFacts['output'], papers['output'], readings['output'], 
                        ytlinks, prev_ai_research)
        research_text = [userInput, intro['output'], quantFacts['output'], papers['output'], readings['output'], 
                            ytlinks, prev_ai_research]
        embedding_function = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(research_text, embedding_function, persist_directory="./chroma_db")
        vectordb.persist()
        st.session_state.embeddings_db = vectordb

class Document:
def __init__(self, content, topic):
        self.page_content = content
        self.metadata = {"Topic": topic}
# Class that represents a document with specific content. Allows you to create a document by provifing both
# the content and the topic when you create an "instance" "Basically another word for a specific object made using 
# the class Document".

def init_ses_states():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("prev_chat_history", [])
    st.session_state.setdefault("embeddings_db", None)
    st.session_state.setdefault('research', None)
    st.session_state.setdefault("prev_research", None)
    st.session_state.setdefault("books", None)
    st.session_state.setdefault("prev_books", None)
# Allows you to keep track of importent research information in the streamlit app. Ex = different clicks and actions.

def main():
# Sets up the streamlit application. 
    st.set_page_config(page_title="Research Bot")
# Page title is setup here. 
    create_research_db()
# This function is called to make a SQL database for storing research data. 
    llm=OpenAI(temperature=0.7)
# Initializes a language model using the OpenAI Application Programming Interface(API). 
# The value for the "temperature" determines the diversity or randomness of the responses.
# # Temperature ranges from 0.0 to 1.0. 0.0 = deterministic and restricted while 1.0 is diverse responses.   
    embedding_function = OpenAIEmbeddings()
# Initializes and embeddings objec. Used for turning text into numbers. Clustering vectors for semantic 
# understanding of similarity of meaning.  
    init_ses_states()
# Initializes session states in the streamlit app that launches when program is ran. 
# Helps store and share data between user interactions. Session states allow you to save information in the 
# conversation with the user so that you can pic it up with the correct context. 
    if os.path.exists("./chroma_db"):
        st.session_state.embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
# This checks if the directory exists. If yes then it starts up chroma database object and uses embeddings function. 
    st.header("GPT-4 LangChain Agents Research Bot")
# Provides a header for streamlit app when it launches.
    deploy_tab, prev_tab = st.tabs(["Generate Research","Previous Research"])
# Creates two new tabs for streamlit app.
    with deploy_tab:
# Creates a context manager for the generate research previously created. All the code indented after is only for 
# the the tab listed as Generate research. 

        userInput = st.text_area(label="User Input")
        #Creates a text area in streamlit the user can interact with. 
        if st.button("Generate Report") and userInput:
            generate_research(userInput)
            # If there is text inside of the text area the user interacted with then the condition becomes true and
            # the generate_research function is called.
        st.subheader("Chat with Data")
        # Creates a subheader in the tab Chat with Data to guide the user to interact with the data. 
        user_message = st.text_input(label="User Message", key="um1")
        #Creates a variable called user message. The key tracks the state of the text input so 
        # streamlit knows/remembers which user message you are interacting with when the user is  user other parts of 
        # the 
        # application.   
        if st.button("Submit Message") and user_message:
        
            memory = ConversationBufferMemory(memory_key="chat_history")
            chatAgent = initialize_agent(tools, 
                                        llm, 
                                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                        verbose=True, 
                                        memory=memory,
                                        )
    with prev_tab:
        st.dataframe(read_research_table())
        selected_input = st.selectbox(label="Previous User Inputs",options=[i for i in read_research_table().user_input])
        if st.button("Render Research") and selected_input:
            with st.expander("Rendered Previous Research",expanded=True):
                selected_df = read_research_table()
                selected_df = selected_df[selected_df.user_input == selected_input].reset_index(drop=True)
                
                st.subheader("User Input:")
                st.write(selected_df.user_input[0])

                st.subheader("Introduction:")
                st.write(selected_df.introduction[0])

                st.subheader("Quantitative Facts:")
                st.write(selected_df.quant_facts[0])

                st.subheader("Previous Related AI Research:")
                st.write(selected_df.prev_ai_research[0])

                st.subheader("Recent Publications:")
                st.write(selected_df.publications[0])

                st.subheader("Recommended Books:")
                st.write(selected_df.books[0])

                st.subheader("YouTube Links:")
                st.write(selected_df.ytlinks[0])

            st.subheader("Chat with Data")
            prev_user_message = st.text_input(label="User Message", key="um2")
            

if __name__ == '__main__':
    load_dotenv()
    main()


