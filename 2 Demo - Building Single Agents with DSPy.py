# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Building Single Agents with DSPy
# MAGIC
# MAGIC This demonstration introduces **DSPy (Declarative Self-improving Python)**, a declarative framework for building modular AI applications. Unlike traditional prompt engineering where you manually craft and refine text prompts, DSPy allows you to build AI systems using structured, modular Python code that is more reliable, maintainable, and portable across different language models.
# MAGIC
# MAGIC This demonstration will focus on 3 DSPy modules:
# MAGIC - `dspy.Predict` (optional)
# MAGIC - `dspy.ChainOfThought` (optional)
# MAGIC - `dspy.ReAct` (required)
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demonstration, you will be able to:
# MAGIC
# MAGIC - Explain what DSPy is and how it differs from traditional prompt engineering
# MAGIC - Configure DSPy to work with Databricks Model Serving endpoints
# MAGIC - Create simple reasoning chains using `dspy.ChainOfThought`
# MAGIC - Define custom tools (functions) that agents can use to interact with data
# MAGIC - Build a ReAct agent using `dspy.ReAct` that combines reasoning and tool execution
# MAGIC - Interpret agent trajectories and understand the reasoning process behind agent decisions
# MAGIC
# MAGIC > Recall we will be using definition: _an AI agent has the ability to view and analyze its environment (plan) and take action (use tools) to achieve a specific goal_
# MAGIC ## Additional Resources
# MAGIC
# MAGIC Here is a non-exhaustive list of additional resources for further reading on the topic of DSPy:
# MAGIC - [DSPy Official Documentation](https://dspy.ai/#programmingnot-promptinglms)
# MAGIC - [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
# MAGIC - [DSPy Discord Community](https://discord.gg/VzS6RHHK6F)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Environment Setup and DSPy Installation
# MAGIC
# MAGIC Before we can use DSPy, we need to install the required libraries and restart the Python environment to ensure all dependencies are properly loaded.
# MAGIC
# MAGIC We'll install a few necessary libraries:
# MAGIC - **[dspy](https://dspy.ai/)**: The core DSPy framework
# MAGIC - **[mlflow](https://www.databricks.com/product/managed-mlflow)**: For model tracking and management (version 3.0+)
# MAGIC - **[pydantic](https://docs.pydantic.dev/latest/)**: For defining structured data models with type validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Compute Requirements
# MAGIC
# MAGIC **🚨 REQUIRED - SELECT SERVERLESS COMPUTE & A SQL WAREHOUSE**
# MAGIC
# MAGIC This course has been configured to run on Serverless compute. While classic compute may also work, testing has been performed on serverless.
# MAGIC
# MAGIC This demo was tested using version 4 of Serverless compute. To ensure that you are using the correct version of Serverless, please navigate to the **Environment** button on the right and open it (see screenshot below).
# MAGIC
# MAGIC ![optional alt text](./Includes/images/serverless-version.png)

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-2

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Enable MLflow Tracing
# MAGIC
# MAGIC Enable automatic tracing with MLflow to capture agent execution details for debugging and analysis.
# MAGIC
# MAGIC **NOTE:** When using a serverless environment, you will need to enable autologging for the MLflow tracing UI to appear as a part of the agent's output. Because MLflow is [integrated with popular GenAI libraries](https://mlflow.org/docs/latest/genai/tracing/#one-line-auto-tracing-integrations), this is actually quite simple to initiate with `mlflow.<framework>.autolog()` as shown in the next cell.

# COMMAND ----------

import mlflow

# Activate MLflow DSPy tracing
mlflow.dspy.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Configure Catalog and Schema
# MAGIC
# MAGIC **🚨 NOTE:** You will need to update the following cell to your catalog name. The schema name will be created automatically for you based on the course name.
# MAGIC
# MAGIC **🚨 NOTE:** If you are using **Vocareum**, your catalog has already been configured for you and is of the form **labuserXXX_XXX**, which matches your Vocareum username. You should set this as your catalog name.
# MAGIC > Example: catalog_name = "labuser31415926_5358979323"
# MAGIC
# MAGIC The catalog and schema variables are used throughout this notebook when referencing Unity Catalog assets.

# COMMAND ----------

# Used when needing to pass catalog/schema name with Python
catalog_name = <FILL_IN>
schema_name = "genai_dspy"
dev_lab_setup(catalog_name, schema_name) # Store the catalog and schema as catalog_name and schema_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### A4. Our Data
# MAGIC This demonstration relies on the Airbnb dataset from Databricks Marketplace. Note that you may already have access to the Airbnb dataset.
# MAGIC #### Vocareum: You have access
# MAGIC If you launched this demo in a Vocareum environemnt, you will automatically have access to the Delta share. It is called `dbacademy_airbnb_sample_data`. For Vocareum users, you will set 
# MAGIC >`databricks_share_name=dbacademy_airbnb_sample_data`. 
# MAGIC #### Non-Vocareum: Check if you have access
# MAGIC Check in the **Catalog Explorer** by searching for `databricks_airbnb_sample_data`. Provided you have the proper level of permisisons on this delta share, you can update the next cell to read 
# MAGIC >`databricks_share_name=databricks_airbnb_sample_data`. 
# MAGIC
# MAGIC ##### I don't have access/can't see the dataset
# MAGIC If you don't have access or can't see the dataset in your **Catalog Explorer**, the next set of instructions will help walk you through how to get this dataset in your workspace.
# MAGIC
# MAGIC 1. Navigate to Marketplace and search **Airbnb Sample Data** and click on the tile that reads **Airbnb Sample Data**.
# MAGIC 1. Next, click **Get instant access** and follow the on-screen instructions to bring that dataset in.
# MAGIC 1. Create a unique Databricks share name. If a name is already in use, you will need to use a different name. Copy the same name into the cell below. For example:
# MAGIC >`databricks_share_name=<unique_name>`. 

# COMMAND ----------

## TODO
databricks_share_name = <FILL_IN> # update to a unique share name you created

# COMMAND ----------

# MAGIC %md
# MAGIC 1. As a part of the classroom setup, a helper function has been configured for you process the dataset from the Delta share. Run the cell to process the CSV `sf-airbnb.csv` from the Airbnb Delta share volume `v01`. 

# COMMAND ----------

df = process_csv(databricks_share_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A5. Create and Register Unity Catalog Functions
# MAGIC
# MAGIC To focus our attention on DSPy, as a part of the classroom setup, we have created a helper function called `tool_creation()` that will use your catalog and schema defined above to create two functions.
# MAGIC
# MAGIC ### Instructions
# MAGIC
# MAGIC 1. Run the next cell to create two tools using the helper function `tool_creation()`. This will create a tool called `avg_neigh_price`
# MAGIC 1. After running the next cell, navigate to the **Catalog Explorer** and search and review the definition and meta information for these functions.
# MAGIC
# MAGIC This will create two SQL functions and store them in Unity Catalog in the schema you provided. Our agent will use these UC functions as tools to answer questions about San Francisco Airbnb listings.

# COMMAND ----------

tool_creation(catalog_name, schema_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Configure DSPy with Databricks Foundation Model Serving API
# MAGIC
# MAGIC DSPy needs to connect to a language model to power its reasoning and decision-making capabilities. In this section, we'll configure DSPy to use a Databricks Foundation Model Serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. Load Configuration Settings
# MAGIC
# MAGIC **NOTE:** This demo assumes you have a configuration file (`demo_agent.json`) that specifies your LLM endpoint and parameters. In production, you might store these in Databricks secrets or environment variables. [Databricks recommends using MLflow's Models from Code as a pattern for logging agents](https://docs.databricks.com/aws/en/generative-ai/agent-framework/log-agent#-code-based-logging), where the agents code is captured as a Python file, and the Python environment is captured as a list of packages. 

# COMMAND ----------

import json

# Load JSON file with LLM configuration
with open("./demo_agent.json", "r") as f:
    config = json.load(f)

llm_endpoint = config['llm_endpoint']

print("Endpoint:", llm_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ### B2. Initialize DSPy Language Model
# MAGIC
# MAGIC Now we create a DSPy language model object (`dspy.LM`) that points to our Databricks endpoint. The `dspy.configure()` function sets this as the default LM for all DSPy operations.
# MAGIC
# MAGIC **Key Concept:** DSPy abstracts away the details of different LLM providers. Whether you use OpenAI, Anthropic, or Databricks, the DSPy code remains largely the same. If you're on the Databricks platform, authentication is automatic via the [Databricks SDK](https://docs.databricks.com/aws/en/dev-tools/sdk-python). If not, you can set the environment variables `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE`, or pass `api_key` and `api_base` below.

# COMMAND ----------

import dspy

lm = dspy.LM(f"databricks/{llm_endpoint}")

dspy.configure(lm=lm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Introduction to DSPy Modules
# MAGIC
# MAGIC A DSPy module is foundational to understanding a DSPy program. Each module abstracts a prompting technique that handles _any_ signature and has _learnable parameters_ to process inputs and return outputs. Composing multiple modules creates a program (inspired by neural network modules in PyTorch). The idea is to parameterize each module so that it can learn its desired behavior.
# MAGIC
# MAGIC We will showcase three modules in this demo:
# MAGIC 1. `dspy.Predict`: This is the most basic and fundamental module
# MAGIC 2. `dspy.ChainOfThought`: More complex than `Predict` in that the system thinks step-by-step before producing an output
# MAGIC 3. `dspy.ReAct`: Combines reasoning and acting capabilities for tool-based interactions
# MAGIC
# MAGIC **NOTE:** If you are familiar with the concept of DSPy modules or want to skip to building a ReAct agent with DSPy, you may skip this section.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. (Optional) Understanding `dspy.Predict`
# MAGIC
# MAGIC All DSPy modules are built on the `dspy.Predict` module, making it the most fundamental module. It stores a supplied _signature_, which we define in the next cell in shorthand (we will define more complex signatures later).
# MAGIC
# MAGIC #### Adapters
# MAGIC In the MLflow trace UI below, you will see the call to `ChatAdapter.format`. We won't go into too many details, but essentially, adapters are configurable (`dspy.configure(adapter=<adapter_type>)`) bridges between the fundamnetal module (`dspy.Predict`) and the LLM/SLM. They are responsible for:
# MAGIC The adapter system is responsible for:
# MAGIC - Translating DSPy signatures into **system messages** that define the task and request/response structure.
# MAGIC - Formatting input data according to the request structure outlined in DSPy signatures.
# MAGIC - Parsing LM responses back into structured DSPy outputs, such as `dspy.Prediction` instances.
# MAGIC - Managing conversation history and function calls.
# MAGIC - Converting pre-built DSPy types into LM prompt messages, e.g., `dspy.Tool`, `dspy.Image`, etc.
# MAGIC > You can read more about adapters [here](https://dspy.ai/learn/programming/adapters/#understanding-dspy-adapters).
# MAGIC
# MAGIC
# MAGIC
# MAGIC ![optional alt text](./Includes/images/dspy_predict.png)

# COMMAND ----------

signature = "question -> answer: str"

# COMMAND ----------

# MAGIC %md
# MAGIC Here, the signature input field is `question` and the output field is `answer` which has a type hint given by `str` (string). When calling `dspy.Predict`, the language model will format a prompt to implement the signature and includes the demonstrations that will then be fed into the LLM/SLM and, eventually, a parsed output.
# MAGIC
# MAGIC Run the next cell and view the **Summary** tab in the MLflow Trace UI. Notice that we see the trace displaying **Input → LM.__call__ → Outputs**, showing that it is the most basic module for a straightforward input-to-output task with no tool calling. We will see that swapping out the `Predict` with `ChainOfThought` will have a different trace in the next section.

# COMMAND ----------

qa_predict = dspy.Predict(signature)

question = "4 dice are tossed. What is the probability that the sum of the dice equals two or 4?"
qa_predict(question = question)

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. (Optional) Understanding `dspy.ChainOfThought`
# MAGIC
# MAGIC In general, Chain of Thought is a prompting technique where the LLM/SLM is encouraged to think step-by-step before providing an answer. With DSPy, you can think of this as an additional reasoning layer on top of the `dspy.Predict` method. However, note that `dspy.ChainOfThought` does not have the capability to use tools.
# MAGIC
# MAGIC Run the next cell with the same signature and question. Note in the **Summary** tab of the MLflow Trace UI, we see the sequence **Input → Predict.forward → LM.__call__ → Outputs**. Hence, we have an additional layer given by `Predict.forward`.
# MAGIC
# MAGIC **Where can you trace the difference for DSPy modules like `dspy.Predict`?** If you expand the `LM.__call__` from the previous cell, you will see that the system prompt is generated so that only `question` and `answer` are provided as input fields to the LLM/SLM call. However, doing the same (expand `LM.__call__`) to the cell below, you can see that there's an additional input field brought in `reasoning`. Also, you can look at the trace of `Predict.forward` call and see that there is a clear separation of **Input** and **Output**, but the **Output** distinguishing **reasoning** from **answer**.

# COMMAND ----------

# Create a Chain of Thought module for math reasoning
qa_cot = dspy.ChainOfThought(signature)

# Ask the question using chain of thought
qa_cot(question=question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Load and Explore the Airbnb Dataset
# MAGIC
# MAGIC Now that we understand some basic terminology, let's load our San Francisco Airbnb listings data to exhibit tool use with `dspy.ReAct`. This dataset contains real Airbnb listings with information about properties, hosts, pricing, availability, and reviews. This will allow us to create a more complex signature and use the `dspy.ReAct` module (recall that in the signature used above, we used the shorthand notation `"question -> answer: str"`).

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Load the Data
# MAGIC
# MAGIC We'll load the data into a Spark DataFrame and then convert a sample to Pandas for easier manipulation in our agent tools.
# MAGIC > We already have `df` defined above, but we read it in from UC below to demonstrate best practices.

# COMMAND ----------

# Load the Airbnb listings table
df = spark.table("sf_airbnb_listings")

# Display basic information
print(f"Total listings: {df.count()}")
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Convert to Pandas for Agent Use
# MAGIC
# MAGIC For this demo, we'll work with a subset of the data in Pandas format. This makes it easier to query and manipulate within our Python-based agent tools.

# COMMAND ----------

def airbnb_pd():
    # Select relevant columns and convert to Pandas
    columns_of_interest = [
        'id', 'name', 'description', 'neighbourhood_cleansed', 
        'property_type', 'room_type', 'accommodates', 'bedrooms', 
        'beds', 'price', 'minimum_nights', 'maximum_nights',
        'number_of_reviews', 'review_scores_rating', 'instant_bookable'
    ]
    df = spark.table("sf_airbnb_listings")

    # Load a sample into Pandas (limiting for demo purposes)
    airbnb_pdf = df.select(columns_of_interest).toPandas()

    # Clean price column (remove $ and convert to float)
    airbnb_pdf['price'] = airbnb_pdf['price'].str.replace('$', '').str.replace(',', '').astype(float)

    print(f"Loaded {len(airbnb_pdf)} listings for agent use")
    return airbnb_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Define Data Structures with Pydantic
# MAGIC
# MAGIC Before creating our agent tools, we need to define the data structures our agent will work with using [**Pydantic**](https://docs.pydantic.dev/latest/#pydantic-validation). We do this using [Pydantic's base class `BaseModel`](https://docs.pydantic.dev/latest/api/base_model/).
# MAGIC
# MAGIC Run the next cell to create a Pydantic data model for listing information with key details about Airbnb properties. The purpose of this step is to tell the agent the type of information our dataset should contain.

# COMMAND ----------

from pydantic import BaseModel
from typing import Optional

class ListingInfo(BaseModel):
    """Represents an Airbnb listing with key details"""
    listing_id: int
    name: str
    description: Optional[str]
    neighborhood: str
    property_type: str
    room_type: str
    accommodates: int
    bedrooms: Optional[int]
    beds: Optional[int]
    price: float
    minimum_nights: int
    review_score: Optional[int]
    number_of_reviews: int

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Define Agent Tools
# MAGIC
# MAGIC Now that we have our data structure, let's create the tools that our agent can use to help users. These tools will interact with our Airbnb dataset.
# MAGIC
# MAGIC **NOTE:** The following is _not_ a UC function (see [the LangChain demo]($./Building Single Agents with LangChain)).

# COMMAND ----------

# MAGIC %md
# MAGIC ### F1. Using Non-UC Tools
# MAGIC
# MAGIC For DSPy ReAct agents, ensure each tool has:
# MAGIC 1. **Have a clear docstring**: Explains what the tool does. Think of this as a prompt to the LLM
# MAGIC - You do not _need_ a docstring, but it is recommended when the function name may not be clear to the LLM/SLM
# MAGIC 2. **Use type hints**: Specifies argument types so the LLM knows what to provide
# MAGIC 3. **Handle errors gracefully**: Return meaningful error messages
# MAGIC
# MAGIC Run the next two cells to define the tool `get_listing_details()` and test the function with `listing_id = 958`.

# COMMAND ----------

def get_listing_details(listing_id: int) -> ListingInfo:
    """
    Get detailed information about a specific listing by its ID.
    """

    airbnb_pdf = airbnb_pd()
    listing_row = airbnb_pdf[airbnb_pdf['id'] == listing_id]
    
    if len(listing_row) == 0:
        raise ValueError(f"Listing with ID {listing_id} not found.")
    
    row = listing_row.iloc[0]
    return ListingInfo(
        listing_id=int(row['id']),
        name=str(row['name']),
        description=str(row['description']) if row['description'] else None,
        neighborhood=str(row['neighbourhood_cleansed']),
        property_type=str(row['property_type']),
        room_type=str(row['room_type']),
        accommodates=int(row['accommodates']),
        bedrooms=int(row['bedrooms']) if row['bedrooms'] else None,
        beds=int(row['beds']) if row['beds'] else None,
        price=float(row['price']),
        minimum_nights=int(row['minimum_nights']),
        review_score=int(row['review_scores_rating']) if row['review_scores_rating'] else None,
        number_of_reviews=int(row['number_of_reviews'])
    )

# COMMAND ----------

get_listing_details(958)

# COMMAND ----------

# MAGIC %md
# MAGIC ### F2. Using UC Tools
# MAGIC
# MAGIC Next, let's read in our UC functions previously created as a part of the classroom setup. In particular, run the next cell to test the function `avg_neigh_price`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT avg_neigh_price('Mission') as average_price

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will define the DSPy tool by wrapping it in a Python function similar to the tool `get_listing_details`. We will use the same SQL query as we ran in the previous cell, which allows us to govern tools via UC while integrating with the DSPy framework. 

# COMMAND ----------

def avg_neigh_price(neighborhood: str):
    """
    Get the average price of listings in a specific neighborhood.
    """

    # This is the same query as defined in the previous code cell. 
    query = f"SELECT avg_neigh_price('{neighborhood}') as average_price"
    result = spark.sql(query).collect()
    if result:
        return result[0]['average_price']
    else:
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC Run the next cell to validate the output is the same as above.

# COMMAND ----------

avg_neigh_price('Mission')

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Build the ReAct Agent
# MAGIC
# MAGIC Now we're ready to build our agent using **DSPy's ReAct** framework.
# MAGIC
# MAGIC ReAct is an agent paradigm that alternates between:
# MAGIC - **Reasoning**: The agent thinks about what to do next and produces reasoning traces
# MAGIC - **Acting**: The agent calls a tool to gather information or perform an action based on observations
# MAGIC
# MAGIC This loop continues until the agent has enough information to answer the user's request.
# MAGIC
# MAGIC > You can read more about the ReAct module [here](https://dspy.ai/api/modules/ReAct/?h=dspy+react#dspyreact). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### G1. Define the Agent Signature
# MAGIC
# MAGIC Keeping in mind this demonstration is an introduction to DSPy, we will need to know how to build one of the core components of a DSPy program, which is a DSPy _signature_. Essentially, a signature allows you to tell an LLM/SLM what it needs to do, rather than how to ask the LLM/SLM to do it. In contrast to frameworks like LangChain that often organize pipelines as chains or components, DSPy uses built-in optimizers for multi-stage logic.
# MAGIC
# MAGIC The idea of a DSPy signature is similar to a function signature in Python, where the types and names of input and output fields are specified, but in DSPy the signature also serves as a contract guiding how the language model should process information. Additionally, you should be mindful of names and semantic roles (e.g., `question` has a semantic difference compared to `answer`).
# MAGIC
# MAGIC Run the next cell to create a Python class that takes in a `dspy.Signature` object and defines a task along with the input and output (`user_request` and `response` as `InputField` and `OutputField`, respectively).
# MAGIC
# MAGIC > The following is known as a **class-based signature** and you can read more about them [here](https://dspy.ai/learn/programming/signatures/#class-based-dspy-signatures).

# COMMAND ----------

class AirbnbRecommendationAgent(dspy.Signature):
    """You are an Airbnb recommendation assistant that helps users find the perfect place to stay. You have access to a database of San Francisco Airbnb listings. You can search for listings, get details about specific properties, analyze neighborhoods, and provide personalized recommendations. Use the available tools to help users find accommodations that match their needs, preferences, and budget. Always be helpful, informative, and provide specific details like listing IDs, prices, and neighborhoods.
    """
    
    user_request: str = dspy.InputField(desc="The user's question or request about Airbnb listings")
    response: str = dspy.OutputField(
        desc="A helpful response that addresses the user's request with specific information, "
             "including listing details, recommendations, or neighborhood insights."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### G2. Configuring the DSPy Agent
# MAGIC
# MAGIC Now we instantiate the agent by providing:
# MAGIC 1. The signature (task definition). Recall we just defined our signature in the previous cell, `AirbnbRecommendationAgent`
# MAGIC 2. The list of tools the agent can use. Recall we will be using two tools: `get_listing_details` and `avg_neigh_price`
# MAGIC
# MAGIC **NOTE:** Even though we have 1 custom tool, `len(agent.tools)` will return a value of 2. That is because the tool `finish` is always equipped. Run the subsequent cell to equip our agent with the custom tool `get_listing_details`.

# COMMAND ----------

agent = dspy.ReAct(
    AirbnbRecommendationAgent,
    tools=[
        get_listing_details,
        avg_neigh_price
    ]
)

print("✅ Airbnb Recommendation Agent created successfully!")
print(f"Agent has access to {len(agent.tools)} tools \n")
print(f"Description of the tool {agent.tools['get_listing_details']}: {agent.tools['get_listing_details'].desc}")
print(f"Description of the tool {agent.tools['finish']}: {agent.tools['finish'].desc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## H. Use the Agent: Example Interactions
# MAGIC
# MAGIC Let's see our agent in action! We'll test it with various user requests to demonstrate its capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### H1. Example Query Using Both Tools
# MAGIC
# MAGIC 1. Run the next cell to ask the agent about listing with ID 958
# MAGIC 2. The MLflow Trace UI will appear in the output. Click on **Summary**. You will see a more complex sequence of reasoning, but more importantly we can see that our tool `get_listing_details()` was used. Note that we can view the output as having multiple layers

# COMMAND ----------

result = agent(
    user_request="Tell me about listing ID 958 and if it's higher than the average price for the Mission neighborhood."
)

print("Agent Response:")
print(result.response)
print("\n" + "="*80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### H2. Inspect Agent Execution History
# MAGIC
# MAGIC DSPy provides powerful introspection capabilities to understand how your agent arrived at its conclusions. It provides `inspect_history()` to see the actual LLM calls and prompts used behind the scenes. This is invaluable for debugging and understanding agent behavior.

# COMMAND ----------

# View the last few LLM interactions
dspy.inspect_history(n=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demonstration, you learned how DSPy's declarative framework simplifies the creation of intelligent, maintainable agents. You saw how Chain-of-Thought reasoning and the ReAct pattern enable agents to plan, reason, and use tools effectively. You also explored the importance of well-designed tools with clear structure and how DSPy's observability features, especially with MLflow—make debugging and understanding agent behavior straightforward. Overall, DSPy provides a more scalable and transparent alternative to traditional prompt engineering.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> |
# MAGIC <a href="https://help.databricks.com/" target="blank">Terms of Use</a> |
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>