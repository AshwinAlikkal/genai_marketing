import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import requests
from openai import OpenAI
from IPython.display import Image
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import prompts
import config
import plotly.graph_objects as go
import streamlit as st


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
    
    # Perform K-means clustering with k clusters
    kmeans = KMeans(n_clusters = config.k, random_state=0)
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

def plot_stacked_bar_chart(df, column, title, yaxis_title, is_channel=False, is_product=False, is_children=False):
    if is_channel:
        cluster_data = df.groupby('Cluster')[column].sum()
    elif is_product:
        cluster_data = df.groupby('Cluster')[column].sum()
    elif is_children:
        cluster_data = df.groupby(['Cluster', column])[column].count().unstack().fillna(0)
    else:
        cluster_data = df.groupby(['Cluster', column])[column].count().unstack().fillna(0)

    cluster_pct = cluster_data.div(cluster_data.sum(axis=1), axis=0) * 100

    fig = go.Figure()

    for col in cluster_pct.columns:
        fig.add_trace(go.Bar(
            x=cluster_pct.index,
            y=cluster_pct[col],
            name=col if not is_children else f'Total Children: {col}'
        ))

    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title='Cluster',
        yaxis_title=yaxis_title,
        legend_title=column if isinstance(column, str) and not is_children else 'Total Children',
        xaxis=dict(tickmode='linear'),
        height=300,  # Adjust plot height if needed
        margin=dict(t=40, b=40)
    )

    for cluster in cluster_pct.index:
        for col in cluster_pct.columns:
            percentage = cluster_pct.loc[cluster, col]
            fig.add_annotation(
                x=cluster,
                y=cluster_pct.loc[cluster, :col].sum() - percentage / 2,
                text=f'{percentage:.1f}%',
                showarrow=False,
                font=dict(size=8, color='white')
            )

    st.plotly_chart(fig, use_container_width=True)

def plot_box_plot(df, column, title, yaxis_title):
    fig = go.Figure()

    clusters = sorted(df['Cluster'].unique())  # Ensure clusters are sorted
    for cluster in clusters:
        cluster_data = df[df['Cluster'] == cluster][column]
        fig.add_trace(go.Box(
            y=cluster_data, 
            name=f'Cluster {cluster}', 
            marker=dict(size=8)  # Increase the size of the box plot markers
        ))

    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        boxmode='group',
        height=300,  # Adjust plot height if needed
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
