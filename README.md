# Lorekeeper

RAG model specializing in *The Lord of the Rings* and *The Hobbit* books by J.R.R Tolkein. 

## Setup

### Download and Install Llama3.2:1b

1. Install Ollama [here](https://ollama.com/).
2. Download *Llama3.2:1b* to local

```
ollama pull llama3.2:1b
```

3. Start *Llama3.2:1b*

```
ollama serve
```

Llama3.2:1b is now being served locally on port **11434**. 

### Install RAG model

1. Create virtual environment

```
python -m venv .lorekeeper
```

2. Install libraries

```
pip install -r requirements.txt
```

3. Create directory in the project folder and name it `data/` and place the PDFs in this directory. 

4. Embed the corpus.

```
python embed.py
```

Confirm that a directory named `embeddings/` has been created which contains `metadata.json` and `vectordata.index`. 

5. Run app

```
streamlit run app.py
```

The RAG model is now running on `localhost:8501/`. 


The RAG model is now ready to use. 