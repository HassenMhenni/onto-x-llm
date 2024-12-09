
# Onto-X LLM

## 1. Project Overview
**Goal**: Use a Large Language Model (LLM) to navigate a pathology ontology called "Onto-X."

### Onto-X dataset ?
Onto-X is a dataset that represents a hierarchy of pathological entities. Each entity has:
- **Class Id**: A unique identifier.
- **Preferred Label**: A human-readable name or label.
- **Parents**: One or more direct ancestors listed, separated by a “|”.

### Main Task
Given an entity’s Preferred Label, reconstruct its full ancestry (all the way to the top-level entities) and determine the depth of each ancestor. This enables a clear, hierarchical view of the ontology.

---

## 2. Project Features and Prompt Explanation

### LLM Integration
We use a the new llama 3.3 70b LLM using groq api(will be explained later)  to interact with the ontology in natural language. The prompt logic is designed to:
- Take a Preferred Label as input.
- Look up the entity in Onto-X.
-  find all its ancestors by following the "Parents" field.
- Assign depths to each ancestor (distance from the queried entity).
- Return the dictionary/table as a table.

### Prompt Explanation
The prompt instructs to the LLM to:
- Only use the provided dataset to find ancestors (no outside knowledge/hallucinations).
- Return a dictionary/table of ancestors with their corresponding depths.
- If the entity is not found or has no parents, return an empty dictionary/table or a predefined message.


## 4. Installation Steps

### Install Dependencies:
```bash
pip install -r requirements.txt
```

## 5. Setting Up the LLM

This project uses the Groq API for integrating Meta's Llama 3.3 70B language model with LangChain. Groq provides high-performance computing solutions and APIs to enable seamless interaction with advanced language models.

### Steps to Set Up

1. **Obtain a Groq API Key**:
   - Visit the [GroqCloud Developer Console](https://console.groq.com/docs/api-keys) to create and manage your API keys.
   - Log in and navigate to the "API Keys" section to generate a new key.

2. **Configure the Environment Variable**:
   - Create a `.env` file in the root directory of your project.
   - Add the following line to the `.env` file, replacing `your_groq_api_key_here` with your API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - This variable is required for authenticating your requests to the Groq API.

3. **Set the Model**:
   - Specify the model ID for Llama 3.3 70B in your API requests. According to Groq's [Supported Models](https://console.groq.com/docs/models), the model ID is:
     ```
     llama-3.3-70b-versatile
     ```
   - Ensure your application uses this model ID to interact with the correct language model.

By completing these steps, your project will be configured to leverage the capabilities of the Llama 3.3 70B model through Groq's API within the LangChain framework.

4. **PandasAI**:

PandasAI is a Python library that enhances pandas dataframes with generative AI capabilities, enabling conversational data analysis.
## 6. Running the App

### Start the Streamlit App:
```bash
streamlit run app.py
```

### Usage:
- The app will open in your browser.
- Input the Preferred Label of the entity you’re interested in.
- Click "Get Ancestors" to retrieve a structured JSON of its ancestors and their depths.

---

## 8. File Structure
- **`app.py`**: Contains the Streamlit front-end interface.
- **`dataset.py`**: Loads the ontology dataset (`onto_x.csv`) into a pandas DataFrame.
- **`llm.py`**: Houses the logic for sending queries to the LLM and processing responses.
- **`onto_x.csv`**: The ontology dataset.
- **`requirements.txt`**: Lists the project dependencies.

---

## 9. Usage Examples

Example inputs and outputs will be provided.

### Screenshots:
![Screenshot of app Interface](screenshots\llm_screenshot.png)

