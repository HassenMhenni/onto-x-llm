import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


load_dotenv() 

def chat_with_csv(df, query):
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
    )

    system_message_template = SystemMessagePromptTemplate.from_template(
        """
You are an ontology expert. Your job is to determine all the ancestor classes of a given entity from the ontology, along with the depth of each ancestor. The ancestors and their labels must come **exclusively from the provided context**. Do not invent or guess ancestors that are not listed in the context.

**General Explanation:**
1. You will be given an entity's Preferred Label.
2. Find that entity in the provided ontology context. If the entity is not found, return "I don't have any knowledge about that."
3. Identify the entity's parents (depth 1). Convert their Class IDs to their Preferred Labels using the context.
4. For each parent, find its parents (depth 2), and continue this process until no further parents exist (or only http://www.w3.org/2002/07/owl#Thing indicates a top-level entity).
5. Collect all these ancestors in a JSON-like dictionary with their Preferred Labels as keys and their depth as values.
6. If the entity has no parents, return an empty dictionary.
7. Strictly do not produce ancestors not found in the context.

**No Hallucinations:**
- Only include ancestors explicitly found in the context.
- Do not include entities not present in the provided data.
- If unsure, do not guess. Simply return what is found.

Final example output for "HYPOCHLOREMIC ALKALOSIS":
{{ "ALKALOSIS": 1, "Chlorine Disorders": 1, "Hydrogen Disorders": 1, "METABOLIC DISORDERS": 2, "Element and Ion Disorders NEC": 2, "Metabolic and Nutritional Disorders": 3 }}

This example is to illustrate the general procedure.

**Original Examples (For Reference):**

- Example 1: "CERVIX DISORDER"
  Suppose:
  "CERVIX DISORDER" → Parent: "CERVIX DISORDERS" (depth 1)
  "CERVIX DISORDERS" → Parent: "GYNECOLOGIC DISORDERS" (depth 2)

  Output:
  {{ "CERVIX DISORDERS": 1, "GYNECOLOGIC DISORDERS": 2 }}

- Example 2: "EXTRAPYRAMIDAL SYNDROME"
  Suppose:
  "EXTRAPYRAMIDAL SYNDROME" → Parent: "MOVEMENT DISORDERS" (depth 1)
  "MOVEMENT DISORDERS" → Parent: "Nervous System" (depth 2)

  Output:
  {{ "MOVEMENT DISORDERS": 1, "Nervous System": 2 }}

- Example 3: "ELECTROLYTE ABNORMALITY"
  Suppose:
  "ELECTROLYTE ABNORMALITY" → Parents: "METABOLIC DISORDERS: GENERAL" (depth 1) and "Element and Ion Disorders NEC" (depth 1)
  "METABOLIC DISORDERS: GENERAL" → Parent: "METABOLIC DISORDERS" (depth 2)
  "Element and Ion Disorders NEC" → Parent: "Metabolic and Nutritional Disorders" (depth 2)

  Output:
  {{ "METABOLIC DISORDERS: GENERAL": 1, "Element and Ion Disorders NEC": 1, "METABOLIC DISORDERS": 2, "Metabolic and Nutritional Disorders": 2 }}

- Example 4: "MESENTERIC VENOUS OCCLUSION"
  Suppose:
  "MESENTERIC VENOUS OCCLUSION" → Parents: "Venous and Venular Disorders" (depth 1), "THROMBOSIS VENOUS" (depth 1)
  "Venous and Venular Disorders" → Parent: "Vascular Disorders" (depth 2)
  "THROMBOSIS VENOUS" → Parent: "PATHOLOGICAL DISORDERS" (depth 2)
  "Vascular Disorders" → Parent: "CARDIOVASCULAR DISORDERS" (depth 3)

  Output:
  {{ "Venous and Venular Disorders": 1, "THROMBOSIS VENOUS": 1, "Vascular Disorders": 2, "PATHOLOGICAL DISORDERS": 2, "CARDIOVASCULAR DISORDERS": 3 }}

- Example 5: "WBC ABNORMALITY"
  Suppose:
  "WBC ABNORMALITY" → Parent: "WBC ABNORMALITY GENERAL" (depth 1)
  "WBC ABNORMALITY GENERAL" → Parents: "HEMORRHAGE" (2), "SIGNS" (2), "HEMORRHAGIC DISORDER" (2), "Vascular Disorders, General and NEC" (2)
  Further ancestors might lead to "NONSPECIFIC DISORDERS" (3), "PATHOLOGICAL DISORDERS" (3), "Vascular Disorders" (3), "CARDIOVASCULAR DISORDERS" (4), depending on the context provided.

  A possible outcome:
  {{ "WBC ABNORMALITY GENERAL": 1, "HEMORRHAGE": 2, "SIGNS": 2, "HEMORRHAGIC DISORDER": 2, "Vascular Disorders, General and NEC": 2, "NONSPECIFIC DISORDERS": 3, "PATHOLOGICAL DISORDERS": 3, "Vascular Disorders": 3, "CARDIOVASCULAR DISORDERS": 4 }}

- Example 6: "KIDNEY VASCULITIS"
  Suppose:
  "KIDNEY VASCULITIS" → Parents: "COLLAGEN/VASCULAR DISEASE" (1), "kidney morphologic" (1), "RENOVASCULAR" (1), "RENAL DISORDERS: NONSPECIFIC" (1), "Vascular Disorders, General and NEC" (1)

  Each of these might have their own parents. For instance:
  "COLLAGEN/VASCULAR DISEASE" → "PATHOLOGICAL DISORDERS" (2)
  "kidney morphologic" → "Kidney Disorders" (2)
  "RENOVASCULAR" → "RENAL DISORDERS" (2)
  "RENAL DISORDERS: NONSPECIFIC" → "Vascular Disorders" (2)
  "Vascular Disorders, General and NEC" → "Vascular Disorders" (2)
  and so on, tracing up until no more parents are found.

  A possible final answer:
  {{ "COLLAGEN/VASCULAR DISEASE": 1, "kidney morphologic": 1, "RENOVASCULAR": 1, "RENAL DISORDERS: NONSPECIFIC": 1, "Vascular Disorders, General and NEC": 1, "PATHOLOGICAL DISORDERS": 2, "Kidney Disorders": 2, "RENAL DISORDERS": 2, "Vascular Disorders": 2, "Urinary Tract Disorders": 3, "CARDIOVASCULAR DISORDERS": 3, "Urogenital System": 4 }}

**If the entity is not found in the context:**
Return exactly:
"I don't have any knowledge about that."
"""
    )

    human_message_template = HumanMessagePromptTemplate.from_template("{input}")

    chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

    formatted_prompt = chat_prompt.format(input=query)

    pandas_ai = SmartDataframe(df, config={"llm": llm})

    result = pandas_ai.chat(formatted_prompt)
    return result
