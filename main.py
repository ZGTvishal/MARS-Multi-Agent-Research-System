import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.state import AgentState
from core.utils import get_corpus_index_path, get_index_path, split_text
from agents.indexing import _model

abstract = "The world is facing a water crisis. The lack of water resources is a major challenge for the " 
# "health and wellbeing of the world's population. This project aims to develop an artificial intelligence based water management system to optimize water resources. The system will provide a platform for water users and water managers to access information, solve problems and make informed decisions for water resource management. Resources of water management involve the supply, allocation and use of water, and the impact of these factors on ecosystems and human well-being. The overall goal of this project is to provide a platform for data to be stored, analyzed, and presented in a user-friendly way. The system will also be able to predict future water resource usage and monitor the status of water resources. The data will be provided by sensors, which will provide the data for the artificial intelligence system. This project will involve the creation of an artificial intelligent system that will be able to predict the future quality of water resources using a combination of data from sensors in the field and artificial intelligence. The artificial intelligent system will be utilized on a water quality management system such as a water treatment plant. The project will also involve the use of artificial intelligence based sensors, which will be capable of detecting water quality and producing a visual representation in the form of graphs on a tablet or PC screen. The data from the sensors will be inputted into the artificial intelligence system."


chunk = split_text(abstract)

print(len(chunk))


# embedding = _model.encode(chunk, convert_to_numpy= True)

# print(embedding.shape)