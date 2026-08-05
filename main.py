from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()


query = ["'title': 'Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture', \n 'abstract': 'We introduce the Momentum Transformer, an attention-based deep-learning architecture, which outperforms benchmark time-series momentum and mean-reversion trading strategies. Unlike state-of-the-art Long Short-Term Memory (LSTM) architectures, which are sequential in nature and tailored to local processing, an attention mechanism provides our architecture with a direct connection to all previous time-steps. Our architecture, an attention-LSTM hybrid, enables us to learn longer-term dependencies, improves performance when considering returns net of transaction costs and naturally adapts to new market regimes, such as during the SARS-CoV-2 crisis. Via the introduction of multiple attention heads, we can capture concurrent regimes, or temporal dynamics, which are occurring at different timescales. The Momentum Transformer is inherently interpretable, providing us with greater insights into our deep-learning momentum trading strategy, including the importance of different factors over time and the past time-steps which are of the greatest significance to the model."]


mod_qry = ['Researchers increasingly rely on machine learning to analyze large collections of scientific literature. A well-designed retrieval pipeline can identify relevant papers, extract meaningful passages,and generate concise summaries without',
'concise summaries without overwhelming the reader. However, the quality of the final output depends heavily on careful preprocessing. Splitting documents into sensible chunks preserves context while keeping inputs within model limits.', 
'within model limits. Overlap between adjacent chunks helps retain continuity across boundaries, reducing the chance that important ideas become separated. Choosing chunk size should be guided by real data rather than', 
'data rather than guesses. Measuring actual abstract lengths from representative samples provides a stronger basis for configuration decisions and makes experiments easier to justify. Consistent evaluation with repeatable test fixtures also', 
'test fixtures also improves confidence when changing retrieval parameters or embedding strategies. Documenting these observations ensures future contributors understand why particular defaults were selected and when they should be revisited later.']

# _model = SentenceTransformer("all-MiniLM-L6-v2")
# encode_query = _model.encode(query, convert_to_numpy=True)

messages = [
    (
        "system",
        "You are a research assistant that will help generate detailed summary of a research paper abstract and title given to you as Human query.",
    ),
    ("human", query),
]


llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=os.getenv("GEMMA_API_KEY"),
        temperature=1.0
    )

raw_response = llm.invoke(messages)
output_text = raw_response.content[-1]
print(output_text)


