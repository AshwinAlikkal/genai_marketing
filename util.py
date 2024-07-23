from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import prompts

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import warnings
warnings.filterwarnings("ignore")


def llm_image_model():
    client = OpenAI()
    return client

def llm_chat_model():  
    llm = ChatOpenAI(model = 'gpt-4o-mini', 
                     temperature = 1
                     )
    return llm


def cluster_creation(df):
    # Columns used for clustering 
    cols_for_clustering = ["Country", 
                           "Education", 
                           "Income", 
                           "Total_Spent", 
                           "Living_With", 
                           "Total_Children",
                            "Is_Parent"]
    df_cluster = df[cols_for_clustering]
    df_cluster = pd.get_dummies(df_cluster, dtype=int)
    
    # Normalizing
    scale_norm = StandardScaler()
    df_cluster_std = pd.DataFrame(scale_norm.fit_transform(df_cluster), columns=df_cluster.columns)
    
    # Perform K-means clustering with 10 clusters
    kmeans = KMeans(n_clusters=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df_cluster_std)
    df["Cluster"] += 1
    return df

def image_generator(prompt, llm):
    response = llm.images.generate(
      model="dall-e-3",
      prompt=prompt,
      size="1792x1024",
      quality="standard",
      style = "natural",
      n=1,
    )
    image_url = response.data[0].url
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def characteristic_prompt_generation(df_cluster, llm):
    system_prompt = prompts.system_prompt_for_characteristic_prompt
    conversation_history = [SystemMessage(content=system_prompt)]
    conversation_history.append(HumanMessage(content=df_cluster.head(10).to_json()))
    response= llm(conversation_history).content
    return response

def stable_diffusion_prompt_generation(characteristic_prompt, llm):
    system_prompt = prompts.system_prompt_for_stable_diffusion
    conversation_history = [SystemMessage(content=system_prompt)]
    conversation_history.append(HumanMessage(content=characteristic_prompt))
    response_stable_diffusion= llm(conversation_history).content
    return response_stable_diffusion
